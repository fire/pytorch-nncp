#
# NNCP v2 (PyTorch version)
#
# Initial version from https://github.com/kimiyoung/transformer-xl
#
# The modifications are (c) 2020 Fabrice Bellard and released under
# the Apache-2.0 License
#
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from mem_transformer import MemTransformerLM
from exp_utils import create_exp_dir
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream

parser = argparse.ArgumentParser(description='NNCP')
parser.add_argument('--decompress', action='store_true',
                    help='decompress input to output instead of compressing')
parser.add_argument('--input', type=str, help='input file')
parser.add_argument('--output', type=str, help='output file')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--lr', type=str,
                    help='learning rate (syntax: lr0[,step1,lr1,...])')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=64,
                    help='number of tokens to predict')
parser.add_argument('--ext_tgt_len', type=int, default=0,
                    help='addional tokens in the prediction context (>= tgt_len)')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--log-interval', type=int, default=100000,
                    help='report interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--vocab_size', type=int, default=256,
                    help='vocabulary size')
parser.add_argument('--adam_eps', type=float, default=1e-8,
                    help='ADAM epsilon')
parser.add_argument('--block_len', type=int, default=500000,
                    help='encoding block length')
parser.add_argument('--gelu', action='store_true',
                    help='use GELU activation instead of ReLU')
parser.add_argument('--tied_r_bias', action='store_true',
                    help='tied r_bias in all the layers')
parser.add_argument('--retrain_period', type=int, default=0,
                    help='retrain period, in symbols, 0 to disable retrain')
parser.add_argument('--retrain_len', type=int, default=10000000,
                    help='retrain length, in symbols')
parser.add_argument('--retrain_batch_size', type=int, default=32,
                    help='retrain batch size')
parser.add_argument('--retrain_tgt_len', type=int, default=64,
                    help='retrain tgt_len')
parser.add_argument('--retrain_mem_len', type=int, default=64,
                    help='retrain mem_len')
parser.add_argument('--retrain_lr', type=str,
                    help='retrain learning rate (syntax: lr0[,step1,lr1,...])')
parser.add_argument('--profiler', action='store_true',
                    help='Enable profiler')

args = parser.parse_args()

if args.d_embed < 0:
    args.d_embed = args.d_model

args.work_dir = '{}-{}'.format(args.work_dir, "enwik8")
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['nncp.py', 'mem_transformer.py'], debug=args.debug)
if args.output is None:
    if args.debug:
        args.output = "/tmp/out.bin"
    else:
        args.output = os.path.join(args.work_dir, "out.bin")
    
# enable deterministic behavior
torch.set_deterministic(True)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            import apex
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m, name):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            if name.find('ff2') != -1:
                nn.init.normal_(m.weight, 0.0, args.init_std * math.sqrt(args.d_model / args.d_inner))
            else:
                init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

model = MemTransformerLM(args.vocab_size, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        d_embed=args.d_embed,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=0, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len,
        tied_r_bias = args.tied_r_bias, use_gelu = args.gelu)

for name, module in model.named_modules():
    weights_init(module, name)

# debug backward pass
def backward_hook(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')

def debug_backward_init(m):
    m.register_backward_hook(backward_hook)

#model.apply(debug_backward_init)

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

args.block_len = (args.block_len // (args.batch_size * args.tgt_len)) * (args.batch_size * args.tgt_len)
if args.retrain_period != 0:
    args.retrain_period = round(args.retrain_period / args.block_len) * args.block_len

model = model.to(device)

def lerp(a, b, t):
    return a + (b - a) * t

def parse_lr_schedule(str):
    lr_tab = []

    p = str.find(",")
    pos = p
    if pos < 0:
        pos = len(str)
    lr = float(str[0:pos])
    lr_tab.append(lr)
    if p >= 0:
        pos = p + 1
        while True:
            p = str.find(",", pos)
            if p < 0:
                break;
            interp_type = 0 # linear interpolation (a*x+b)
            if str[pos] == 'p':
                interp_type = 1 # power interpolation (b*x^a)
                pos += 1
            lr_tab.append(interp_type)
            step = int(str[pos:p])
            lr_tab.append(step)
            pos = p + 1
            p = str.find(",", pos)
            if p < 0:
                p = len(str)
            lr = float(str[pos:p])
            lr_tab.append(lr)
            pos = p + 1
    return lr_tab

def eval_lr_schedule(step, lr_tab):
    start_lr = lr_tab[0]
    i = 1
    start_step = 0
    while i < len(lr_tab):
        interp_type = lr_tab[i]
        end_step = lr_tab[i + 1]
        end_lr = lr_tab[i + 2]
        if step <= end_step:
            if interp_type == 0:
                return lerp(start_lr, end_lr, (step - start_step) / (end_step - start_step))
            else:
                a = math.log(end_lr/start_lr)/math.log(end_step/start_step)
                return start_lr * ((step / start_step) ** a)
        start_step = end_step
        start_lr = end_lr
        i += 3
    return start_lr

#### optimizer & scheduler
lr_schedule = parse_lr_schedule(args.lr)
if args.retrain_period != 0:
    retrain_lr_schedule = parse_lr_schedule(args.retrain_lr)
    
if args.fp16:
    adam_opt = apex.optimizers.FusedAdam
else:
    adam_opt = optim.Adam

optimizer = adam_opt(model.parameters(), lr=lr_schedule[0], betas=(0, 0.9999), eps = args.adam_eps, weight_decay=0.0)

if args.retrain_period != 0:
    retrain_optimizer_state = optimizer.state_dict()

if args.fp16:
    # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2")


    
logging('=' * 80)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 80)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

header_disp = 0

def show_header():
    global header_disp
    if header_disp == 0:
        logging("M     STEP       SIZE      CSIZE locBPS    BPS   kS/s       LR");
        header_disp = (header_disp + 1) % 20
    
retrain_train_step = 0

def retrain(file_data, file_pos):
    global retrain_train_step, retrain_optimizer_state

    saved_optimizer_state = optimizer.state_dict()
    optimizer.load_state_dict(retrain_optimizer_state)
    
    model.train() # enable dropout
    model.reset_length(args.retrain_tgt_len, 0, args.retrain_mem_len)
    tgt_len = args.retrain_tgt_len
    batch_size = args.retrain_batch_size
    block_len = min(file_pos, args.retrain_len)
    block_start = file_pos - block_len
    last_time = time.time()
    
    block_stride = block_len // batch_size
    stream_len = block_stride
    stream_pos = 0
    mems = tuple()

    last_disp_pos = 0
    stream_pos = 0
    last_n_bits = 0
    n_bits = 0
    while (stream_pos + tgt_len) <= stream_len:
        # build the data and target tensors
        target0 = []
        data0 = []
        for j in range(0, tgt_len):
            pos = block_start + stream_pos + j
            target0 += file_data[pos: pos + block_stride * batch_size: block_stride]

            if stream_pos == 0:
                data1 = [0] * batch_size # dummy input
            else:
                pos = block_start + stream_pos + j - 1
                data1 = file_data[pos: pos + block_stride * batch_size: block_stride]             
            data0.append(data1)
            
        data = torch.LongTensor(data0).to(device)
        target = torch.LongTensor(target0).to(device)
            
        # apply the model
        
        optimizer.zero_grad()
        ret = model(data, tgt_len, *mems)
        output, mems = ret[0], ret[1:]

        logit = F.log_softmax(output, -1)
        logit = torch.reshape(logit, (tgt_len * batch_size, logit.size(2)))
        loss = F.nll_loss(logit, target, reduction='none')
        
        n_bits += loss.float().sum().item() / math.log(2)
        loss = loss.float().mean().type_as(loss)
        if args.fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer),
                                           args.clip)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # simpler than an explicit LR scheduler
        optimizer.param_groups[0]['lr'] = eval_lr_schedule(retrain_train_step, retrain_lr_schedule)
            
        optimizer.step()
        
        n_syms = (stream_pos - last_disp_pos) * batch_size
        if n_syms >= args.log_interval:
            cur_time = time.time()
            log_str = "R {:8d} {:10d} {:>10s} {:6.3f} {:>6s} {:6.2f} {:8.3g}". \
                      format(retrain_train_step,
                             block_start + stream_pos * batch_size,
                             "-",
                             (n_bits - last_n_bits) / n_syms,
                             "-",
                             n_syms / ((cur_time - last_time) * 1000.0),
                             optimizer.param_groups[0]['lr'])
            logging(log_str)
            last_disp_pos = stream_pos
            last_n_bits = n_bits
            last_time = cur_time
        
        retrain_train_step += 1
        stream_pos += tgt_len
    model.eval() # disable dropout
    model.reset_length(args.tgt_len, 0, args.mem_len)
    retrain_optimizer_state = optimizer.state_dict()
    optimizer.load_state_dict(saved_optimizer_state)

# return a list
def load_file_be16(filename):
    print("loading " + filename);
    f = open(filename, "rb")
    byte_buffer = f.read();
    f.close();
    data = []
    for i in range(0, len(byte_buffer), 2):
      data.append(byte_buffer[i] * 256 + byte_buffer[i + 1])
    return data

def compute_retrain_max_step(file_len):
    n_retrain = file_len // args.retrain_period
    step = 0
    for i in range(n_retrain):
        block_len = min(args.retrain_len, (i + 1) * args.block_len)
        step += block_len // (args.retrain_tgt_len * args.retrain_batch_size)
    return step
                                  
def train():
    decode_flag = args.decompress

    out_file = open(args.output, "wb")
    if not decode_flag:
        file_data = load_file_be16(args.input)
        original_file_len = len(file_data)
        bit_output = BitOutputStream(out_file)
        arith_enc = ArithmeticEncoder(32, bit_output)
        out_file.write(original_file_len.to_bytes(4, byteorder='big'))
    else:
        assert(args.tgt_len == 1)
        in_file = open(args.input, "rb")
        original_file_len = int.from_bytes(in_file.read(4), byteorder='big')
        file_data = [0] * original_file_len
        bit_input = BitInputStream(in_file)
        arith_dec = ArithmeticDecoder(32, bit_input)

    tgt_len = args.tgt_len
    ext_tgt_len = args.ext_tgt_len
    batch_size = args.batch_size

    # pad the file with zeros so that its size is a multiple of
    # batch_size * tgt_len (negligible loss for large files)
    d = batch_size * tgt_len
    file_len = ((original_file_len + d - 1) // d) * d
    file_pad_len = file_len - original_file_len
    file_data += [0] * file_pad_len
    
    logging("file_length = {}, max_step = {}".format(original_file_len, file_len // (batch_size * tgt_len)))
    if args.retrain_period != 0:
        logging("retrain_max_step = {}".format(compute_retrain_max_step(file_len)))
    
    # round to an integer number steps
    file_len = (file_len // (batch_size * tgt_len)) * (batch_size * tgt_len)

    plot_file = None
    if not args.debug:
        plot_filename = os.path.join(args.work_dir, 'plot.txt')
        logging("plot filename : {}".format(plot_filename))
        plot_file = open(plot_filename, 'w')
    
    model.eval() # disable dropout
    n_input_symbs = 0
    last_n_input_symbs = 0
    n_output_bits = 0.0
    last_n_output_bits = 0.0
    start_time = time.time()
    last_time = start_time
    train_step = 0
    block_start = 0
    last_retrain_pos = 0
    while block_start < file_len:
        block_len = min(file_len - block_start, args.block_len)

        # retrain 
        if args.retrain_period != 0 and block_start - last_retrain_pos >= args.retrain_period:
            retrain(file_data, block_start)
            last_retrain_pos = block_start

        block_stride = block_len // batch_size
        stream_len = block_stride
        stream_pos = 0
        mems = tuple()
        while (stream_pos + tgt_len) <= stream_len:
            # build the tensor
            data0 = []
            for j in range(0, tgt_len + ext_tgt_len):
                pos = stream_pos - ext_tgt_len + j - 1
                if pos < 0:
                    data1 = [0] * batch_size # dummy input
                else:
                    pos += block_start
                    data1 = file_data[pos: pos + block_stride * batch_size: block_stride]
                data0.append(data1)
            data = torch.LongTensor(data0).to(device)

            # apply the model

            optimizer.zero_grad()
            ret = model(data, tgt_len, *mems)
            output, mems = ret[0], ret[1:]

            if tgt_len == 1:
                prob = F.softmax(output.detach(), -1)
                # convert to cumulative frequency table
                freq = torch.round(prob * 10000000).int()
                freq = torch.max(freq, freq.new_ones(freq.size()))
                freq = torch.cumsum(freq, -1)
                freq = freq.cpu()
                
                target0 = []
                for j in range(tgt_len):
                    for i in range(batch_size):
                        pos = block_start + stream_pos + j + i * block_stride
                        freq_tab = freq[j][i]
                        if decode_flag:
                            sym = arith_dec.read(freq_tab)
                            file_data[pos] = sym
                        else:
                            sym = file_data[pos]
                            arith_enc.write(freq_tab, sym)
                        target0.append(sym)
            else:
                # faster version for testing without arithmetic coding
                target0 = []
                for j in range(tgt_len):
                    pos = block_start + stream_pos + j
                    target0 += file_data[pos: pos + block_stride * batch_size: block_stride]
                
            # compute the loss
            target = torch.LongTensor(target0).to(device)
            logit = F.log_softmax(output, -1)
            logit = torch.reshape(logit, (tgt_len * batch_size, logit.size(2)))
            loss = F.nll_loss(logit, target, reduction='none')
            
            n_output_bits += loss.float().sum().item() / math.log(2)
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer),
                                               args.clip)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # simpler than an explicit LR scheduler
            optimizer.param_groups[0]['lr'] = eval_lr_schedule(train_step, lr_schedule)
            optimizer.step()

            # statistics
            n_input_symbs += batch_size * tgt_len
            is_last_step = stream_pos + tgt_len >= stream_len and block_start + block_len >= file_len
            if n_input_symbs - last_n_input_symbs >= args.log_interval or is_last_step:
                show_header()
                cur_time = time.time()
                log_str = "  {:8d} {:10d} {:10.0f} {:6.3f} {:6.3f} {:6.2f} {:8.3g}". \
                          format(train_step,
                                 n_input_symbs,
                                 n_output_bits / 8,
                                 (n_output_bits - last_n_output_bits) / (n_input_symbs - last_n_input_symbs),
                                 n_output_bits / n_input_symbs,
                                 (n_input_symbs - last_n_input_symbs) / ((cur_time - last_time) * 1000.0),
                                 optimizer.param_groups[0]['lr'])
                logging(log_str)
                if plot_file is not None:
                    plot_file.write("{:10d} {:10.0f}\n".format(n_input_symbs,
                                                           n_output_bits / 8))
                    plot_file.flush()
                last_n_input_symbs = n_input_symbs
                last_n_output_bits = n_output_bits
                last_time = cur_time
                
            train_step += 1
            stream_pos += tgt_len
            
        if decode_flag:
            for j in range(block_start, min(block_start + block_len, original_file_len)):
                out_file.write(file_data[j].to_bytes(2, byteorder='big'))
            out_file.flush()

        block_start += block_len
        
    if not decode_flag:
        arith_enc.finish()
        bit_output.close()
    out_file.close()
        
    if plot_file is not None:
        plot_file.close()

    total_time = time.time() - start_time
    logging("Total time {:0.1f} s ({:0.2f} kS/s)".\
            format(total_time,
                   n_input_symbs / (total_time * 1000.0)));
                                                       
### main

if args.profiler:
    with profiler.profile(record_shapes=False) as prof:
        train()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
else:
    train()

