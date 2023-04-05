NNCP v2: Lossless Data Compression with Transformer
===================================================

1) Introduction
---------------

NNCP v2 is a lossless data compressor based on the Transformer
model. It is mostly optimized to compress enwik9 (first GB of the
English Wikipedia).

Read the accompanying article for more information.

2) Installation
---------------

It was tested on the CentOS 8 Linux distribution. A desktop PC was
used with an NVIDIA RTX 3090 GPU. Other GPUs should work provided
there is enough GPU RAM (6 GB are required). CPU only usage may be
possible but much slower.

Type "make" to compile the "preprocess" program (NNCP preprocessor).

The main program "nncp.py" was tested with:

- Python 3.6
- PyTorch 1.7.0
- cuda 11.0
- apex version 8cf5ae61beff5738c87150b6c4348603eeb159d5 from
  https://github.com/NVIDIA/apex. It is required only when using the
  --fp16 option to enable 16 bit floating point support for faster
  operation.

3) Usage
--------

Download enwik8 and enwik9 from http://mattmahoney.net/dc/textdata.html

Then use the following commands:

- nncp_enwik_base.sh c enwik8 (enwik8 compression, lasts 9 hours)

- nncp_enwik_base.sh d enwik8 (enwik8 decompression, lasts 9 hours)

- nncp_enwik_base.sh c enwik9 (enwik9 compression, lasts 3.7 days)

- nncp_enwik_base.sh d enwik9 (enwik9 decompression, lasts 3.7 days)

Warning: Unlike NNCP v1, the compressor has a deterministic output
only when using the same hardware and installed software. So the
compressed output cannot usually be decompressed on a different
machine.

nncp_enwik_large.sh gives better results at the expense of a longer
processing time (about twice slower).

4) License
----------

The "preprocess" source code is released under the MIT license.

nncp.py is based on the source code from:

- https://github.com/kimiyoung/transformer-xl (Transformer XL reference
model, Apache 2.0 license)

- https://github.com/nayuki/Reference-arithmetic-coding (arithmetic coder, MIT License)

- https://github.com/byronknoll/tensorflow-compress (simplifications
  in the arithmetic coder, public domain).
