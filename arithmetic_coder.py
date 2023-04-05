#@title Arithmetic Coding Library

# 
# Reference arithmetic coding
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import sys
python3 = sys.version_info.major >= 3


# ---- Arithmetic coding core classes ----

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase(object):
	
	# Constructs an arithmetic coder, which initializes the code range.
	def __init__(self, numbits):
		if numbits < 1:
			raise ValueError("State size out of range")
		
		# -- Configuration fields --
		# Number of bits for the 'low' and 'high' state variables. Must be at least 1.
		# - Larger values are generally better - they allow a larger maximum frequency total (maximum_total),
		#   and they reduce the approximation error inherent in adapting fractions to integers;
		#   both effects reduce the data encoding loss and asymptotically approach the efficiency
		#   of arithmetic coding using exact fractions.
		# - But larger state sizes increase the computation time for integer arithmetic,
		#   and compression gains beyond ~30 bits essentially zero in real-world applications.
		# - Python has native bigint arithmetic, so there is no upper limit to the state size.
		#   For Java and C++ where using native machine-sized integers makes the most sense,
		#   they have a recommended value of num_state_bits=32 as the most versatile setting.
		self.num_state_bits = numbits
		# Maximum range (high+1-low) during coding (trivial), which is 2^num_state_bits = 1000...000.
		self.full_range = 1 << self.num_state_bits
		# The top bit at width num_state_bits, which is 0100...000.
		self.half_range = self.full_range >> 1  # Non-zero
		# The second highest bit at width num_state_bits, which is 0010...000. This is zero when num_state_bits=1.
		self.quarter_range = self.half_range >> 1  # Can be zero
		# Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
		self.minimum_range = self.quarter_range + 2  # At least 2
		# Maximum allowed total from a frequency table at all times during coding. This differs from Java
		# and C++ because Python's native bigint avoids constraining the size of intermediate computations.
		self.maximum_total = self.minimum_range
		# Bit mask of num_state_bits ones, which is 0111...111.
		self.state_mask = self.full_range - 1
		
		# -- State fields --
		# Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
		self.low = 0
		# High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
		self.high = self.state_mask
	
	
	# Updates the code range (low and high) of this arithmetic coder as a result
	# of processing the given symbol with the given frequency table.
	# Invariants that are true before and after encoding/decoding each symbol
	# (letting full_range = 2^num_state_bits):
	# - 0 <= low <= code <= high < full_range. ('code' exists only in the decoder.)
	#   Therefore these variables are unsigned integers of num_state_bits bits.
	# - low < 1/2 * full_range <= high.
	#   In other words, they are in different halves of the full range.
	# - (low < 1/4 * full_range) || (high >= 3/4 * full_range).
	#   In other words, they are not both in the middle two quarters.
	# - Let range = high - low + 1, then full_range/4 < minimum_range
	#   <= range <= full_range. These invariants for 'range' essentially
	#   dictate the maximum total that the incoming frequency table can have.
	def update(self, freqs, symbol):
		# State check
		low = self.low
		high = self.high
		# if low >= high or (low & self.state_mask) != low or (high & self.state_mask) != high:
		# 	raise AssertionError("Low or high out of range")
		range = high - low + 1
		# if not (self.minimum_range <= range <= self.full_range):
		# 	raise AssertionError("Range out of range")
		
		# Frequency table values check
		total = int(freqs[-1])
		symlow = int(freqs[symbol-1]) if symbol > 0 else 0
		symhigh = int(freqs[symbol])
		#total = freqs.get_total()
		#symlow = freqs.get_low(symbol)
		#symhigh = freqs.get_high(symbol)
		# if symlow == symhigh:
		# 	raise ValueError("Symbol has zero frequency")
		# if total > self.maximum_total:
		# 	raise ValueError("Cannot code symbol because total is too large")
		
		# Update range
		newlow  = low + symlow  * range // total
		newhigh = low + symhigh * range // total - 1
		self.low = newlow
		self.high = newhigh
		
		# While low and high have the same top bit value, shift them out
		while ((self.low ^ self.high) & self.half_range) == 0:
			self.shift()
			self.low  = ((self.low  << 1) & self.state_mask)
			self.high = ((self.high << 1) & self.state_mask) | 1
		# Now low's top bit must be 0 and high's top bit must be 1
		
		# While low's top two bits are 01 and high's are 10, delete the second highest bit of both
		while (self.low & ~self.high & self.quarter_range) != 0:
			self.underflow()
			self.low = (self.low << 1) ^ self.half_range
			self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1
	
	
	# Called to handle the situation when the top bit of 'low' and 'high' are equal.
	def shift(self):
		raise NotImplementedError()
	
	
	# Called to handle the situation when low=01(...) and high=10(...).
	def underflow(self):
		raise NotImplementedError()


# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):
	
	# Constructs an arithmetic coding encoder based on the given bit output stream.
	def __init__(self, numbits, bitout):
		super(ArithmeticEncoder, self).__init__(numbits)
		# The underlying bit output stream.
		self.output = bitout
		# Number of saved underflow bits. This value can grow without bound.
		self.num_underflow = 0
	
	
	# Encodes the given symbol based on the given frequency table.
	# This updates this arithmetic coder's state and may write out some bits.
	def write(self, freqs, symbol):
		self.update(freqs, symbol)
	
	
	# Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
	# It is important that this method must be called at the end of the each encoding process.
	# Note that this method merely writes data to the underlying output stream but does not close it.
	def finish(self):
		self.output.write(1)
	
	
	def shift(self):
		bit = self.low >> (self.num_state_bits - 1)
		self.output.write(bit)
		
		# Write out the saved underflow bits
		for _ in range(self.num_underflow):
			self.output.write(bit ^ 1)
		self.num_underflow = 0
	
	
	def underflow(self):
		self.num_underflow += 1


# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):
	
	# Constructs an arithmetic coding decoder based on the
	# given bit input stream, and fills the code bits.
	def __init__(self, numbits, bitin):
		super(ArithmeticDecoder, self).__init__(numbits)
		# The underlying bit input stream.
		self.input = bitin
		# The current raw code bits being buffered, which is always in the range [low, high].
		self.code = 0
		for _ in range(self.num_state_bits):
			self.code = self.code << 1 | self.read_code_bit()
	
	
	# Decodes the next symbol based on the given frequency table and returns it.
	# Also updates this arithmetic coder's state and may read in some bits.
	def read(self, freqs):
		#if not isinstance(freqs, CheckedFrequencyTable):
		#	freqs = CheckedFrequencyTable(freqs)
		
		# Translate from coding range scale to frequency table scale
		total = int(freqs[-1])
		#total = freqs.get_total()
		#if total > self.maximum_total:
		#	raise ValueError("Cannot decode symbol because total is too large")
		range = self.high - self.low + 1
		offset = self.code - self.low
		value = ((offset + 1) * total - 1) // range
		#assert value * range // total <= offset
		#assert 0 <= value < total
		
		# A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
		start = 0
		end = len(freqs)
		#end = freqs.get_symbol_limit()
		while end - start > 1:
			middle = (start + end) >> 1
			low = int(freqs[middle-1]) if middle > 0 else 0
			#if freqs.get_low(middle) > value:
			if low > value:
				end = middle
			else:
				start = middle
		#assert start + 1 == end
		
		symbol = start
		#assert freqs.get_low(symbol) * range // total <= offset < freqs.get_high(symbol) * range // total
		self.update(freqs, symbol)
		#if not (self.low <= self.code <= self.high):
		#	raise AssertionError("Code out of range")
		return symbol
	
	
	def shift(self):
		self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()
	
	
	def underflow(self):
		self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()
	
	
	# Returns the next bit (0 or 1) from the input stream. The end
	# of stream is treated as an infinite number of trailing zeros.
	def read_code_bit(self):
		temp = self.input.read()
		if temp == -1:
			temp = 0
		return temp


# ---- Bit-oriented I/O streams ----

# A stream of bits that can be read. Because they come from an underlying byte stream,
# the total number of bits is always a multiple of 8. The bits are read in big endian.
class BitInputStream(object):
	
	# Constructs a bit input stream based on the given byte input stream.
	def __init__(self, inp):
		# The underlying byte stream to read from
		self.input = inp
		# Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
		self.currentbyte = 0
		# Number of remaining bits in the current byte, always between 0 and 7 (inclusive)
		self.numbitsremaining = 0
	
	
	# Reads a bit from this stream. Returns 0 or 1 if a bit is available, or -1 if
	# the end of stream is reached. The end of stream always occurs on a byte boundary.
	def read(self):
		if self.currentbyte == -1:
			return -1
		if self.numbitsremaining == 0:
			temp = self.input.read(1)
			if len(temp) == 0:
				self.currentbyte = -1
				return -1
			self.currentbyte = temp[0] if python3 else ord(temp)
			self.numbitsremaining = 8
		assert self.numbitsremaining > 0
		self.numbitsremaining -= 1
		return (self.currentbyte >> self.numbitsremaining) & 1
	
	
	# Reads a bit from this stream. Returns 0 or 1 if a bit is available, or raises an EOFError
	# if the end of stream is reached. The end of stream always occurs on a byte boundary.
	def read_no_eof(self):
		result = self.read()
		if result != -1:
			return result
		else:
			raise EOFError()
	
	
	# Closes this stream and the underlying input stream.
	def close(self):
		self.input.close()
		self.currentbyte = -1
		self.numbitsremaining = 0


# A stream where bits can be written to. Because they are written to an underlying
# byte stream, the end of the stream is padded with 0's up to a multiple of 8 bits.
# The bits are written in big endian.
class BitOutputStream(object):
	
	# Constructs a bit output stream based on the given byte output stream.
	def __init__(self, out):
		self.output = out  # The underlying byte stream to write to
		self.currentbyte = 0  # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
		self.numbitsfilled = 0  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
	
	
	# Writes a bit to the stream. The given bit must be 0 or 1.
	def write(self, b):
		if b not in (0, 1):
			raise ValueError("Argument must be 0 or 1")
		self.currentbyte = (self.currentbyte << 1) | b
		self.numbitsfilled += 1
		if self.numbitsfilled == 8:
			towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
			self.output.write(towrite)
			self.currentbyte = 0
			self.numbitsfilled = 0
	
	
	# Closes this stream and the underlying output stream. If called when this
	# bit stream is not at a byte boundary, then the minimum number of "0" bits
	# (between 0 and 7 of them) are written as padding to reach the next byte boundary.
	def close(self):
		while self.numbitsfilled != 0:
			self.write(0)
		self.output.close()
                
