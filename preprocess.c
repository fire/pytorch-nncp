/*
 * NNCP preprocessor
 * 
 * Copyright (c) 2018-2019 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <getopt.h>

#include "cutils.h"

void *mallocz(int size)
{
    void *ptr;
    ptr = malloc(size);
    if (!ptr)
        return NULL;
    memset(ptr, 0, size);
    return ptr;
}

/****************************************************************/

typedef uint16_t DataSymbol;

typedef struct Word {
    uint32_t next; /* -1 = end */
    uint32_t freq;
    float score;
    uint32_t len;
    DataSymbol buf[2];
} Word;

typedef struct {
    Word *words;
    size_t word_count;
    size_t word_size;
    uint32_t *hash_table;
    int hash_size;
    int hash_bits;
} WordList;

static uint32_t hash_calc(const DataSymbol *buf, int len, int n_bits)
{
    uint32_t h;
    int i;

    h = 1;
    for(i = 0; i < len; i++) {
        h = h * 314159 + buf[i];
    }
    return h & ((1 << n_bits) - 1);
}

static void hash_resize(WordList *s, int hash_bits)
{
    int i, h;
    Word *p;
    
    s->hash_bits = hash_bits;
    s->hash_size = 1 << hash_bits;
    free(s->hash_table);
    s->hash_table = malloc(sizeof(s->hash_table[0]) * s->hash_size);
    for(i = 0; i < s->hash_size; i++)
        s->hash_table[i] = -1;
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        h = hash_calc(p->buf, p->len, s->hash_bits);
        p->next = s->hash_table[h];
        s->hash_table[h] = i;
    }
}

static WordList *word_list_init(void)
{
    WordList *s;
    
    s = mallocz(sizeof(WordList));
    s->word_count = 0;
    s->word_size = 0;
    hash_resize(s, 12);
    return s;
}

static void word_list_end(WordList *s)
{
    free(s->words);
    free(s->hash_table);
    free(s);
}

int64_t hash_lookup_count;
int64_t hash_it_count;

/* the hash size contains HASH_SIZE_FACTOR times more entries */
#define HASH_SIZE_FACTOR 2

static Word *word_find_add(WordList *s, const DataSymbol *buf, int len, int add)
{
    uint32_t h, idx;
    Word *p;
    int i;

    assert(len >= 1);
    h = hash_calc(buf, len, s->hash_bits);
    idx = s->hash_table[h];
    hash_lookup_count++;
    while (idx != -1) {
        hash_it_count++;
        p = &s->words[idx];
        if (p->len == len && !memcmp(p->buf, buf, len * sizeof(buf[0])))
            return p;
        idx = p->next;
    }

    if (!add)
        return NULL;

    if (s->word_count >= s->word_size) {
        size_t new_size = s->word_size + s->word_size / 2;
        if (new_size < 32)
            new_size = 32;
        if (s->word_count + 1 > new_size)
            new_size = s->word_count + 1;
        s->words = realloc(s->words, new_size * sizeof(s->words[0]));
        s->word_size = new_size;

    }
    /* resize the hash table when needed */
    if ((s->word_count * HASH_SIZE_FACTOR) > s->hash_size) {
        int hash_bits = s->hash_bits;
        while ((s->word_count * HASH_SIZE_FACTOR) > (1 << hash_bits))
            hash_bits++;
        hash_resize(s, hash_bits);
        
        /* recompute the hash with the new hash table size */
        h = hash_calc(buf, len, s->hash_bits);
    }

    idx = s->word_count++;
    p = &s->words[idx];
    p->freq = 0;
    p->len = len;
    for(i = 0; i < len; i++)
        p->buf[i] = buf[i];
    p->next = s->hash_table[h];
    s->hash_table[h] = idx;
    return p;
}

/****************************************************************/

typedef struct {
    uint32_t len;
    char data[0];
} StringEntry;

typedef struct {
    StringEntry **tab;
    size_t size;
    size_t count;
} StringTable;

StringTable *string_table_init(void)
{
    StringTable *s;
    s = mallocz(sizeof(*s));
    return s;
}

void string_table_add(StringTable *s, const char *data, uint32_t data_len)
{
    size_t new_size;
    StringEntry **new_tab, *se;

    if ((s->count + 1) > s->size) {
        new_size = s->size * 3 / 2;
        if (new_size < s->count + 1)
            new_size = s->count + 1;
        new_tab = realloc(s->tab, sizeof(s->tab[0]) * new_size);
        s->tab = new_tab;
        s->size = new_size;
    }
    se = malloc(sizeof(StringEntry) + data_len + 1);
    se->len = data_len;
    memcpy(se->data, data, data_len);
    se->data[data_len] = '\0';
    s->tab[s->count++] = se;
}

void string_table_end(StringTable *s)
{
    size_t i;
    for(i = 0; i < s->count; i++) {
        free(s->tab[i]);
    }
    free(s->tab);
    free(s);
}

/****************************************************************/

//#define USE_CUT /* disable multiple word combining */

#define MAX_WORDS_PER_ITER 100
#define SUBST_COST 7.0 /* in bits */
#define TOT_FREQ_RED_BITS 1.3 /* log2(old_tot_freq/new_tot_freq) */

#define CH_NO_SPACE     1
#define CH_TO_UPPER     2
#define CH_FIRST_UPPER  3

/* separate words */
#define CH_CUT          0xffffffff

/* number of reserved symbols */
#define NS 256

void dump_word(FILE *f, WordList *s, uint32_t code, BOOL text_output)
{
    Word *p;
    
    if (code < NS) {
        if (text_output) {
            switch(code) {
            case '\n':
                fprintf(f, "\\n");
                break;
            case '\\':
                fprintf(f, "\\%c", code);
                break;
            case CH_TO_UPPER:
                fprintf(f, "\\u");
                break;
            case CH_FIRST_UPPER:
                fprintf(f, "\\c");
                break;
            case CH_NO_SPACE:
                fprintf(f, "\\s");
                break;
            default:
                fprintf(f, "%c", code);
                break;
            }
        } else {
            switch(code) {
            case '\n':
                fprintf(f, "\\n");
                break;
            case '\\':
                fprintf(f, "\\%c", code);
                break;
            default:
                fprintf(f, "%c", code);
                break;
            }
        }
    } else {
        code -= NS;
        assert(code < s->word_count);
        p = &s->words[code];
        dump_word(f, s, p->buf[0], text_output);
        dump_word(f, s, p->buf[1], text_output);
    }
}

typedef struct {
    WordList *s;
    uint32_t *char_freq;
} SortState;

static int word_freq_cmp2(const void *a1, const void *a2, void *arg)
{
    SortState *ss = arg;
    uint32_t c1 = *(DataSymbol *)a1;
    uint32_t c2 = *(DataSymbol *)a2;
    uint32_t freq1, freq2;

    if (c1 < NS)
        freq1 = ss->char_freq[c1];
    else
        freq1 = ss->s->words[c1 - NS].freq;

    if (c2 < NS)
        freq2 = ss->char_freq[c2];
    else
        freq2 = ss->s->words[c2 - NS].freq;

    if (freq1 < freq2)
        return 1;
    else if (freq1 == freq2)
        return 0;
    else
        return -1;
}

#if defined(_WIN32)

static void *rqsort_arg;
static int (*rqsort_cmp)(const void *, const void *, void *);

static int rqsort_cmp2(const void *p1, const void *p2)
{
    return rqsort_cmp(p1, p2, rqsort_arg);
}

/* not reentrant, but not needed with emscripten */
void rqsort(void *base, size_t nmemb, size_t size,
            int (*cmp)(const void *, const void *, void *),
            void *arg)
{
    rqsort_arg = arg;
    rqsort_cmp = cmp;
    qsort(base, nmemb, size, rqsort_cmp2);
}

#else

void rqsort(void *base, size_t n, size_t elem_size,
            int (*cmp)(const void *, const void *, void *),
            void *arg)
{
    qsort_r(base, n, elem_size, cmp, arg);
}

#endif /* !_WIN32 */

int sort_words(WordList *s, uint32_t **ptab, uint32_t *char_freq,
               BOOL sort_by_freq)
{
    uint32_t *tab, n_words;
    int i, j;
    SortState ss_s, *ss = &ss_s;

    /* sort the words */
    n_words = NS + s->word_count;
    tab = malloc(sizeof(tab[0]) * n_words);
    j = 0;
    for(i = 0; i < n_words; i++) {
        if (i >= NS && s->words[i - NS].freq == 0)
            continue;
        tab[j++] = i;
    }
    if (sort_by_freq) {
        ss->s = s;
        ss->char_freq = char_freq;
        rqsort(tab, j, sizeof(tab[0]), word_freq_cmp2, ss);
    }
    *ptab = tab;
    return j;
}

void save_words_debug(WordList *s, const char *filename,
                      const uint32_t *char_freq, uint32_t tot_freq,
                      const uint32_t *tab, int word_count)
{
    FILE *f;
    int i;
    uint32_t c, sum, freq;
    Word *p;
    
    f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    fprintf(f, "%7s %5s %s\n",
            "FREQ", "CUM%", "WORD");
    sum = 0;
    for(i = 0; i < word_count; i++) {
        c = tab[i];
        if (c < NS) {
            freq = char_freq[c];
        } else {
            p = &s->words[c - NS];
            freq = p->freq;
        }
        sum += freq;
        fprintf(f, "%7u %5.1f '", freq, (double)sum / tot_freq * 100);
        dump_word(f, s, c, TRUE);
        fprintf(f, "'\n");
    }
    
    fclose(f);
}

void save_words(WordList *s, const char *filename,
                const uint32_t *tab, int word_count)
{
    FILE *f;
    int i;
    
    f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    for(i = 0; i < word_count; i++) {
        dump_word(f, s, tab[i], FALSE);
        fprintf(f, "\n");
    }
    
    fclose(f);
}

void fput_be16(FILE *f, uint16_t v)
{
    fputc(v >> 8, f);
    fputc(v >> 0, f);
}

int fget_be16(FILE *f, uint16_t *pv)
{
    uint8_t buf[2];
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
        return -1;
    *pv = (buf[0] << 8) |
        (buf[1] << 0);
    return 0;
}

void dump_word_bin(FILE *f, WordList *s, uint32_t *convert_table,
                    uint32_t c)
{
    Word *p;
    
    if (c < NS) {
        fput_be16(f, convert_table[c]);
    } else {
        c -= NS;
        assert(c < s->word_count);
        p = &s->words[c];
        if (p->freq == 0) {
            dump_word_bin(f, s, convert_table, p->buf[0]);
            dump_word_bin(f, s, convert_table, p->buf[1]);
        } else {
            fput_be16(f, convert_table[c + NS]);
        }
    }
}

void save_output(const DataSymbol *buf, size_t buf_len, WordList *s,
                 const char *out_filename, const uint32_t *tab, int word_count)
{
    FILE *fo;
    uint32_t *convert_table;
    size_t i;
    
    fo = fopen(out_filename, "wb");
    if (!fo) {
        perror(out_filename);
        exit(1);
    }

    /* build the convertion table */
    convert_table = malloc(sizeof(convert_table[0]) * (s->word_count + NS));
    for(i = 0; i < s->word_count + NS; i++)
        convert_table[i] = -1;
    for(i = 0; i < word_count; i++) {
        convert_table[tab[i]] = i;
    }

    for(i = 0; i < buf_len; i++) {
        if (buf[i] != CH_CUT) {
            dump_word_bin(fo, s, convert_table, buf[i]);
        }
    }
    free(convert_table);
    
    fclose(fo);
}

static int word_score_cmp(const void *a1, const void *a2)
{
    const Word *p1 = a1;
    const Word *p2 = a2;

    if (p1->score > p2->score)
        return -1;
    else if (p1->score == p2->score)
        return 0;
    else
        return 1;
}

static float get_n_bits(int c, WordList *s,
                        const uint32_t *char_freq, uint32_t tot_freq)
{
    Word *p;
    if (c < NS) {
        return -log2((double)char_freq[c] / tot_freq);
    } else {
        p = &s->words[c - NS];
        if (p->freq == 0) {
            /* deleted word */
            return get_n_bits(p->buf[0], s, char_freq, tot_freq) +
                get_n_bits(p->buf[1], s, char_freq, tot_freq);
        } else {
            return -log2((double)p->freq / tot_freq);
        }
    }
}

static float compute_score(const Word *p, WordList *cw,
                           const uint32_t *char_freq, uint32_t tot_freq)
{
    float old_bits, new_bits;

    if (p->freq <= 1)
        return -1; /* not interesting if not repeating */
    if (1) {
        old_bits = (get_n_bits(p->buf[0], cw, char_freq, tot_freq) +
                    get_n_bits(p->buf[1], cw, char_freq, tot_freq)) * p->freq;
        
        new_bits = (-log2((double)p->freq / tot_freq) + TOT_FREQ_RED_BITS) *
            p->freq + SUBST_COST;
        /* return the gain in bits per substitution */
        return old_bits - new_bits;
    } else {
        return p->freq;
    }
}

/* select at most 'n' non overlaping words in ws and add them in
   cw */
int select_best_words(WordList *s, int n, WordList *cw,
                      const uint32_t *char_freq, uint32_t tot_freq,
                      int min_word_freq)
{
    int i, j;
    Word *p;
    uint8_t *cw_bitmap_start, *cw_bitmap_end;
    const DataSymbol *buf;

    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        p->score = compute_score(p, cw, char_freq, tot_freq);
    }
    
    qsort(s->words, s->word_count, sizeof(s->words[0]), word_score_cmp);

#if 0
    {
        printf("%3s %7s %7s %s\n",
               "N", "FREQ", "SCORE", "WORD");
        for(i = 0; i < min_int(s->word_count, 50); i++) {
            p = &s->words[i];
            printf("%3u %7u %7.0f '", i, p->freq, p->score);
            dump_word(stdout, cw, p->buf[0], TRUE);
            dump_word(stdout, cw, p->buf[1], TRUE);
            printf("'\n");
        }
    }
#endif
    
    cw_bitmap_start = mallocz(NS + cw->word_count);
    cw_bitmap_end = mallocz(NS + cw->word_count);
    
    j = 0;
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        if (p->score <= 0 || p->freq < min_word_freq)
            break;
        /* test if there is a potential overlap with an existing word */
        buf = p->buf;
        if (cw_bitmap_end[buf[0]] ||
            cw_bitmap_start[buf[1]])
            continue;
        cw_bitmap_start[buf[0]] = 1;
        cw_bitmap_end[buf[1]] = 1;
        
        /* add the word */
        word_find_add(cw, buf, 2, TRUE);
#if 0
        printf("%3u %7u '", j, p->freq);
        dump_word(stdout, cw, NS + cw->word_count - 1, TRUE);
        printf("'\n");
#endif
        if (++j >= n)
            break;
    }

    free(cw_bitmap_start);
    free(cw_bitmap_end);
    return j;
}

static void buf_realloc(DataSymbol **pbuf, size_t *pbuf_size, size_t new_size)
{
    if (new_size <= *pbuf_size)
        return;
    new_size = max_size_t(new_size, *pbuf_size + (*pbuf_size) / 8);
    *pbuf = realloc(*pbuf, new_size * sizeof(**pbuf));
    *pbuf_size = new_size;
}

static void out_word(DataSymbol **pbuf, size_t *pbuf_size,
                     size_t *pbuf_pos, WordList *s, uint32_t c)
{
    size_t pos;
    Word *p;
    
    if (c < NS) {
        goto out_char;
    } else {
        p = &s->words[c - NS];
        if (p->freq == 0) {
            out_word(pbuf, pbuf_size, pbuf_pos, s, p->buf[0]);
            out_word(pbuf, pbuf_size, pbuf_pos, s, p->buf[1]);
        } else {
        out_char:
            pos = *pbuf_pos;
            if (pos >= *pbuf_size)
                buf_realloc(pbuf, pbuf_size, pos + 1);
            (*pbuf)[pos++] = c;
            *pbuf_pos = pos;
        }
    }
}

static void compute_word_freq(WordList *s, uint32_t *char_freq,
                              const DataSymbol *buf, size_t buf_size)
{
    int i;
    uint32_t c;
    Word *p;
    
    /* compute the frequency of all the words */
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        p->freq = 0;
    }
    for(i = 0; i < NS; i++) {
        char_freq[i] = 0;
    }
    for(i = 0; i < buf_size; i++) {
        c = buf[i];
        if (c != CH_CUT) {
            if (c >= NS) {
                p = &s->words[c - NS];
                p->freq++;
            } else {
                char_freq[c]++;
            }
        }
    }
}

/* compute the frequency of the words and remove the ones with a too
   low frequency. Return the word count */
static int update_word_freq(WordList *s, uint32_t *char_freq,
                            DataSymbol **pbuf, size_t *pbuf_size,
                            int min_word_freq)
{
    int i, word_count;
    Word *p;
    DataSymbol *obuf, *buf;
    size_t buf_size, obuf_size, buf_pos;
    
    buf_size = *pbuf_size;
    buf = *pbuf;

    compute_word_freq(s, char_freq, buf, buf_size);
    
    word_count = 0;
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        if (p->freq >= min_word_freq) {
            word_count++;
        } else {
            p->freq = 0;
        }
    }
    if (word_count == s->word_count)
        return word_count;
    /* remove the words with a too low score from the buffer */
    obuf = malloc(sizeof(obuf[0]) * buf_size);
    obuf_size = buf_size;
    buf_pos = 0;
    for(i = 0; i < buf_size; i++) {
        out_word(&obuf, &obuf_size, &buf_pos, s, buf[i]);
    }
    free(buf);

    /* update the frequencies */
    compute_word_freq(s, char_freq, obuf, buf_pos);
    
    *pbuf = obuf;
    *pbuf_size = buf_pos;
    return word_count;
}

static double compute_entropy(WordList *s, uint32_t *char_freq,
                              size_t buf_size)
{
    double n_bits;
    size_t i;
    Word *p;
    
    n_bits = 0;
    for(i = 0; i < NS; i++) {
        if (char_freq[i] != 0)
            n_bits += -log2((double)char_freq[i] / buf_size) * char_freq[i];
    }
    for(i = 0; i < s->word_count; i++) {
        p = &s->words[i];
        if (p->freq > 0) {
            n_bits += -log2((double)p->freq / buf_size) * p->freq;
        }
    }
    return n_bits;
}

    
/* CH_WORD_START: add a space except if the last char is '[' or '(' */

static int is_word_char(int c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 128);
}

static int is_upper(int c)
{
    return (c >= 'A' && c <= 'Z');
}

static int is_lower(int c)
{
    return (c >= 'a' && c <= 'z') || (c >= 128);
}

/*
  syntax:
  CH_NO_SPACE?[CH_TO_UPPER|CH_FIRST_UPPER]?SPACE
*/

DataSymbol *case_space_encoding(size_t *pobuf_len, DataSymbol *buf, size_t buf_len)
{
    size_t i, j, len, k, l, obuf_size;
    DataSymbol *obuf;
    int ch_type, c;
    BOOL has_space;
    
    obuf = malloc(sizeof(buf[0]) * buf_len);
    obuf_size = buf_len;
    k = 0;
    for(i = 0; i < buf_len;) {
        if (is_word_char(buf[i])) {
            j = i + 1;
            if (is_lower(buf[i])) {
                while (j < buf_len && is_lower(buf[j]))
                    j++;
                ch_type = 0;
            } else if (j < buf_len && is_upper(buf[j])) {
                while (j < buf_len && is_upper(buf[j]))
                    j++;
                ch_type = CH_TO_UPPER;
            } else {
                while (j < buf_len && is_lower(buf[j]))
                    j++;
                ch_type = CH_FIRST_UPPER;
            }
            len = j - i;

            if (k == 0) {
                has_space = TRUE;
            } else if (obuf[k - 1] == ' ') {
                has_space = TRUE;
                k--;
            } else {
                has_space = FALSE;
            }
            buf_realloc(&obuf, &obuf_size, k + len + 3);
#ifdef USE_CUT
            obuf[k++] = CH_CUT;
#endif
            if (!has_space) {
                obuf[k++] = CH_NO_SPACE;
            }
            if (ch_type != 0)
                obuf[k++] = ch_type;
            obuf[k++] = ' ';
            for(l = 0; l < len; l++) {
                c = buf[i + l];
                if (c >= 'A' && c <= 'Z')
                    c = c - 'A' + 'a';
                obuf[k++] = c;
            }
            i += len;
        } else {
            buf_realloc(&obuf, &obuf_size, k + 1);
            obuf[k++] = buf[i++];
        }
    }
    obuf = realloc(obuf, sizeof(obuf[0]) * k);
    *pobuf_len = k;
    return obuf;
}

void word_encode(const char *in_filename, const char *out_filename,
                 const char *word_filename, int n_words, int min_word_freq,
                 const char *debug_dict_filename, BOOL sort_by_freq)
{
    FILE *f;
    size_t buf_size, i, j, buf_size1;
    DataSymbol *buf, *buf1;
    uint32_t *char_freq, *tab_sort;
    WordList *ws, *s;
    Word *p;
    int n, word_count, word_count_prev;
    double n_bits;
    
    f = fopen(in_filename, "rb");
    if (!f) {
        perror(in_filename);
        exit(1);
    }
    
    fseek(f, 0, SEEK_END);
    buf_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc(buf_size * sizeof(buf[0]));
    for(i = 0; i < buf_size; i++) {
        buf[i] = fgetc(f);
    }
    fclose(f);

    printf("input: %d bytes\n", (int)buf_size);

    buf1 = case_space_encoding(&buf_size1, buf, buf_size);
    free(buf);
    buf = buf1;
    buf_size = buf_size1;
        
    printf("after case/space preprocessing: %d symbols\n",
           (int)buf_size);
    
    s = word_list_init();

    char_freq = mallocz(sizeof(char_freq[0]) * NS);

    compute_word_freq(s, char_freq, buf, buf_size);

    n_words -= NS;
    
    printf("%6s %9s %9s\n",
           "#WORDS", "LENGTH", "BYTES");
    for(word_count = 0; word_count < n_words;) {
        if (buf_size < 2)
            break;
        ws = word_list_init();
        hash_lookup_count = 0;
        hash_it_count = 0;
        for(i = 0; i < buf_size - 1; i++) {
            /* favorise words with space before */
            if (buf[i] != CH_CUT && buf[i + 1] != CH_CUT) {
                p = word_find_add(ws, buf + i, 2, TRUE);
                p->freq++;
            }
        }
#if 0
        printf("hash stats: %d %" PRId64 "d avg=%0.1f\n",
               (int)ws->word_count,
               hash_lookup_count,
               (double)hash_it_count / hash_lookup_count);
#endif
        n = select_best_words(ws, min_int(MAX_WORDS_PER_ITER,
                                          n_words - word_count),
                              s, char_freq, buf_size, min_word_freq);
        word_list_end(ws);
        if (n == 0)
            break;
        
        /* replace with the new words */
        j = 0;
        for(i = 0; i < buf_size;) {
            if ((i + 1) >= buf_size)
                goto no_subst;
            p = word_find_add(s, buf + i, 2, FALSE);
            if (p) {
                buf[j++] = NS + (p - s->words);
                i += 2;
            } else {
            no_subst:
                buf[j++] = buf[i++];
            }
        }
        buf_size = j;

        /* compute the frequency of all the words and remove the words
           with a too low frequency */
        word_count_prev = word_count;
        word_count = update_word_freq(s, char_freq, &buf, &buf_size,
                                      min_word_freq);

        /* mesure the entropy */
        n_bits = compute_entropy(s, char_freq, buf_size);
        
        printf("%6u %9u %9.0f\r", word_count + NS, (int)buf_size, n_bits / 8);
        fflush(stdout);

        if (word_count >= n_words ||
            word_count == word_count_prev)
            break;
    }

    printf("\nNumber of words=%d Final length=%d\n",
           (int)word_count + NS, (int)buf_size);

    word_count = sort_words(s, &tab_sort, char_freq, sort_by_freq);
    
    save_words(s, word_filename, tab_sort, word_count);
    if (debug_dict_filename)
        save_words_debug(s, debug_dict_filename, char_freq, buf_size,
                         tab_sort, word_count);
    
    save_output(buf, buf_size, s, out_filename, tab_sort, word_count);
    free(tab_sort);
    free(buf);
    free(char_freq);
    
    word_list_end(s);
}

void word_load(StringTable *s, const char *filename)
{
    FILE *f;
    uint8_t buf[4096];
    int len, c;
    
    f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(1);
    }
    len = 0;
    for(;;) {
        c = fgetc(f);
        if (c < 0)
            break;
        if (c == '\n') {
            if (len > 0) {
                string_table_add(s, (const char *)buf, len);
            }
            len = 0;
        } else {
            if (c == '\\') {
                c = fgetc(f);
                if (c < 0)
                    break;
                if (c == 'n') {
                    c = '\n';
                } else if (c != '\\') {
                    fprintf(stderr, "Invalid escape\n");
                    exit(1);
                }
            }
            if (len >= sizeof(buf)) {
                fprintf(stderr, "Word too long\n");
                exit(1);
            }
            buf[len++] = c;
        }
    }
    fclose(f);
}

void word_decode(const char *in_filename, const char *out_filename,
                 const char *word_filename)
{
    StringTable *s;
    FILE *f, *fo;
    uint16_t c;
    int i, ch_type, ch_type1, len;
    StringEntry *p;
    const uint8_t *buf;
    BOOL has_space;
    
    s = string_table_init();
    word_load(s, word_filename);

    printf("%d words\n", (int)s->count);
    
    f = fopen(in_filename, "rb");
    if (!f) {
        perror(in_filename);
        exit(1);
    }
    
    fo = fopen(out_filename, "wb");
    if (!fo) {
        perror(out_filename);
        exit(1);
    }

    ch_type = 0;
    has_space = FALSE;
    ch_type1 = 0;
    for(;;) {
        if (fget_be16(f, &c))
            break;
        if (c >= s->count) {
            fprintf(stderr, "Invalid symbol %d\n", c);
            exit(1);
        }
        p = s->tab[c];
        buf = (uint8_t *)p->data;
        len = p->len;
        for(i = 0; i < len; i++) {
            c = buf[i];
            if (c == CH_TO_UPPER || c == CH_FIRST_UPPER) {
                ch_type = c;
            } else if (c == CH_NO_SPACE) {
                has_space = FALSE;
            } else if (c == ' ') {
                ch_type1 = ch_type;
                ch_type = 0;
                if (has_space) {
                    fputc(' ', fo);
                }
                has_space = TRUE;
            } else {
                if (ch_type1 == CH_TO_UPPER || ch_type1 == CH_FIRST_UPPER) {
                    if (c >= 'a' && c <= 'z')
                        c = c - 'a' + 'A';
                    if (ch_type1 == CH_FIRST_UPPER)
                        ch_type1 = 0;
                }
                fputc(c, fo);
            }
        }
    }

    fclose(fo);

    fclose(f);
    
    string_table_end(s);
}


void help(void)
{
    printf("Preprocess version " CONFIG_VERSION", Copyright (c) 2018-2019 Fabrice Bellard\n"
           "Dictionary based preprocessor\n"
           "usage: preprocess [options] c dictfile infile outfile n_words min_freq\n"
           "       preprocess [options] d dictfile infile outfile\n"
           "\n"
           "'c' command: build the dictionary 'dictfile' from 'infile' and output the preprocessed data to 'outfile'. 'n_words' is the approximative maximum number of words of the dictionary. 'min_freq' is the minimum frequency of the selected words.\n"
           "'d' command: rebuild the original file from the dictionary and the preprocessed data.\n"
           "\n"
           "Options:\n"
           "-h             this help\n"
           "-D filename    output debug information associated with the dictionary\n"
           "-s             sort the words by decreasing frequency\n"

);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *in_filename, *out_filename, *mode, *dict_filename;
    const char *debug_dict_filename;
    BOOL sort_by_freq;
    int c;

    debug_dict_filename = NULL;
    sort_by_freq = FALSE;
    for(;;) {
        c = getopt(argc, argv, "hD:s");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help();
            break;
        case 'D':
            debug_dict_filename = optarg;
            break;
        case 's':
            sort_by_freq = TRUE;
            break;
        default:
            exit(1);
        }
    }

    if ((argc - optind) < 1)
        help();

    mode = argv[optind++];

    if (mode[0] == 'c') {
        int n_words, min_word_freq;
        if ((argc - optind) != 5)
            help();
        dict_filename = argv[optind++];
        in_filename = argv[optind++];
        out_filename = argv[optind++];
        n_words = atoi(argv[optind++]);
        min_word_freq = atoi(argv[optind++]);
        word_encode(in_filename, out_filename, dict_filename, n_words,
                    min_word_freq, debug_dict_filename, sort_by_freq);
    } else if (mode[0] == 'd') {
        if ((argc - optind) != 3)
            help();
        dict_filename = argv[optind++];
        in_filename = argv[optind++];
        out_filename = argv[optind++];
        word_decode(in_filename, out_filename, dict_filename);
    } else {
        help();
    }
    return 0;
}

