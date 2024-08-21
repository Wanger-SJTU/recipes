//
//    FILE: float16.cpp
//  AUTHOR: Rob Tillaart
// VERSION: 0.3.0
// PURPOSE: library for Float16s for Arduino
//     URL: http://en.wikipedia.org/wiki/Half-precision_floating-point_format


#include "float16.h"

#include <cmath>

//  CONSTRUCTOR
float16::float16(double f)
{
  _value = f32tof16(f);
}

//////////////////////////////////////////////////////////
//
//  CONVERTING & PRINTING
//
double float16::toDouble() const
{
  return f16tof32(_value);
}

float float16::toFloat() const
{
  return f16tof32(_value);
}

std::string float16::toString(unsigned int decimals) const
{
  return std::to_string((double)f16tof32(_value));
}


//////////////////////////////////////////////////////////
//
//  EQUALITIES
//
bool float16::operator == (const float16 &f)
{
  return (_value == f._value);
}

bool float16::operator != (const float16 &f)
{
  return (_value != f._value);
}

bool float16::operator > (const float16 &f)
{
  if ((_value & 0x8000) && ( f._value & 0x8000)) return _value < f._value;
  if (_value & 0x8000) return false;
  if (f._value & 0x8000) return true;
  return _value > f._value;
}

bool float16::operator >= (const float16 &f)
{
  if ((_value & 0x8000) && (f._value & 0x8000)) return _value <= f._value;
  if (_value & 0x8000) return false;
  if (f._value & 0x8000) return true;
  return _value >= f._value;
}

bool float16::operator < (const float16 &f)
{
  if ((_value & 0x8000) && (f._value & 0x8000)) return _value > f._value;
  if (_value & 0x8000) return true;
  if (f._value & 0x8000) return false;
  return _value < f._value;
}

bool float16::operator <= (const float16 &f)
{
  if ((_value & 0x8000) && (f._value & 0x8000)) return _value >= f._value;
  if (_value   & 0x8000) return true;
  if (f._value & 0x8000) return false;
  return _value <= f._value;
}


//////////////////////////////////////////////////////////
//
//  NEGATION
//
float16 float16::operator - ()
{
  float16 f16;
  f16.setBinary(_value ^ 0x8000);
  return f16;
}


//////////////////////////////////////////////////////////
//
//  MATH
//

namespace 
{

#define SIGN_MASK 0x8000
#define EXP_MASK 0x7C00
#define NAN_VALUE 0x7FFF
#define IS_ZERO(x) (((x) & 0x7FFF) == 0)
#define IS_INVALID(x) (((x) & EXP_MASK) == EXP_MASK)
#define IS_NAN(x) (((x) & 0x7FFF) > 0x7C00)
#define IS_INF(x) ( ((x) & 0x7FFF) == 0x7C00)
#define MANTISSA(x) (((x) & 1023) | (((x) & 0x7C00) == 0 ? 0 : 1024))
#define EXPONENT(x) (((x) & 0x7C00) >> 10)
#define SIGNED_INF_VALUE(x)  ((x & SIGN_MASK) | 0x7C00)

short f16_sub(short ain,short bin);
short f16_add(short ain,short bin);

short f16_sub(short ain,short bin)
{
    unsigned short a=ain;
    unsigned short b=bin;
    if(((a ^ b) & 0x8000) != 0)
        return f16_add(a,b ^ 0x8000);
    unsigned short sign = a & 0x8000;
    a = a << 1;
    b = b << 1;
    if(a < b) {
        unsigned short x=a;
        a=b;
        b=x;
        sign ^= 0x8000;
    }
    unsigned short ax = a & 0xF800;
    unsigned short bx = b & 0xF800;
    if(a >=0xf800 || b>=0xf800) {
        if(a > 0xF800 || b > 0xF800 || a==b)
            return 0x7FFF; 
        unsigned short res = sign | 0x7C00;
        if(a == 0xf800)
            return res;
        else
            return res ^ 0x8000;
    }
    int exp_diff = ax - bx;
    unsigned short exp_part  = ax;
    if(exp_diff != 0) {
        int shift = exp_diff >> 11;
        if(bx != 0)
            b = ((b & 2047) | 2048) >> shift;
        else
            b >>= (shift - 1);
    }
    else {
        if(bx == 0) {
            unsigned short res = (a-b) >> 1;
            if(res == 0)
                return res;
            return res | sign;
        }
        else {
            b=(b & 2047) | 2048;
        }
    }
    unsigned short r=a - b;
    if((r & 0xF800) == exp_part) {
        return (r>>1) | sign;
    }
    unsigned short am = (a & 2047) | 2048;
    unsigned short new_m = am - b;
    if(new_m == 0)
        return 0;
    while(exp_part !=0 && !(new_m & (2048))) {
        exp_part-=0x800;
        if(exp_part!=0)
            new_m<<=1;
    }
    return (((new_m & 2047) | exp_part) >> 1) | sign;
}

short f16_add(short a,short b)
{
    if (((a ^ b) & 0x8000) != 0)
        return f16_sub(a,b ^ 0x8000);
    short sign = a & 0x8000;
    a &= 0x7FFF;
    b &= 0x7FFF;
    if(a<b) {
        short x=a;
        a=b;
        b=x;
    }
    if(a >= 0x7C00 || b>=0x7C00) {
        if(a>0x7C00 || b>0x7C00)
            return 0x7FFF;
        return 0x7C00 | sign;
    }
    short ax = (a & 0x7C00);
    short bx = (b & 0x7C00);
    short exp_diff = ax - bx;
    short exp_part = ax;
    if(exp_diff != 0) {
        int shift = exp_diff >> 10;
        if(bx != 0)
            b = ((b & 1023) | 1024) >> shift;
        else
            b >>= (shift - 1);
    }
    else {
        if(bx == 0) {
            return (a + b) | sign;
        }
        else {
            b=(b & 1023) | 1024;
        }
    }
    short r=a+b;
    if ((r & 0x7C00) != exp_part) {
        unsigned short am = (a & 1023) | 1024;
        unsigned short new_m = (am + b) >> 1;
        r =( exp_part + 0x400) | (1023 & new_m);
    }
    if((unsigned short)r >= 0x7C00u) {
        return sign | 0x7C00;
    }
    return r | sign;
}

short f16_mul(short a,short b)
{
    int sign = (a ^ b) & SIGN_MASK;

    if(IS_INVALID(a) || IS_INVALID(b)) {
        if(IS_NAN(a) || IS_NAN(b) || IS_ZERO(a) || IS_ZERO(b))
            return NAN_VALUE;
        return sign | 0x7C00;
    }

    if(IS_ZERO(a) || IS_ZERO(b))
        return 0;
    unsigned short m1 = MANTISSA(a);
    unsigned short m2 = MANTISSA(b);

    uint32_t v=m1;
    v*=m2;
    int ax = EXPONENT(a);
    int bx = EXPONENT(b);
    ax += (ax==0);
    bx += (bx==0);
    int new_exp = ax + bx - 15;
    
    if(v & ((uint32_t)1<<21)) {
        v >>= 11;
        new_exp++;
    }
    else if(v & ((uint32_t)1<<20)) {
        v >>= 10;
    }
    else { // denormal
        new_exp -= 10;
        while(v >= 2048) {
            v>>=1;
            new_exp++;
        }
    }
    if(new_exp <= 0) {
        v>>=(-new_exp + 1);
        new_exp = 0;
    }
    else if(new_exp >= 31) {
        return SIGNED_INF_VALUE(sign);
    }
    return (sign) | (new_exp << 10) | (v & 1023);
}

short f16_div(short a,short b)
{
    short sign = (a ^ b) & SIGN_MASK;
    if(IS_NAN(a) || IS_NAN(b) || (IS_INVALID(a) && IS_INVALID(b)) || (IS_ZERO(a) && IS_ZERO(b)))
        return 0x7FFF;
    if(IS_INVALID(a) || IS_ZERO(b))
        return sign | 0x7C00;
    if(IS_INVALID(b))
        return 0;
    if(IS_ZERO(a))
        return 0;

    unsigned short m1 = MANTISSA(a);
    unsigned short m2 = MANTISSA(b);
    uint32_t m1_shifted = m1;
    m1_shifted <<= 10;
    uint32_t v= m1_shifted / m2;
    unsigned short rem = m1_shifted % m2;
    
    int ax = EXPONENT(a);
    int bx = EXPONENT(b);
    ax += (ax==0);
    bx += (bx==0);
    int new_exp = ax - bx + 15 ;

    if(v == 0 && rem==0)
        return 0;

    while(v < 1024 && new_exp > 0) {
        v<<=1;
        rem<<=1;
        if(rem >= m2) {
            v++;
            rem -= m2;
        }
        new_exp--;
    }
    while(v >= 2048) {
        v>>=1;
        new_exp++;
    }
    
    if(new_exp <= 0) {
        v>>=(-new_exp + 1);
        new_exp = 0;
    }
    else if(new_exp >= 31) {
        return SIGNED_INF_VALUE(sign);
    }
    return sign | (v & 1023) | (new_exp << 10);
}

short f16_neg(short v)
{
    return SIGN_MASK ^ v;
}
short f16_from_int(int32_t sv)
{
    uint32_t v;
    int sig = 0;
    if(sv < 0) {
        v=-sv;
        sig=1;
    }
    else
        v=sv;
    if(v==0)
        return 0;
    int e=25;
    while(v >= 2048) {
        v>>=1;
        e++;
    }
    while(v<1024) {
        v<<=1;
        e--;
    }
    if(e>=31)
        return SIGNED_INF_VALUE(sig << 15);
    return (sig << 15) | (e << 10) | (v & 1023);
}
int32_t f16_int(short a)
{
    unsigned short value = MANTISSA(a);
    short shift = EXPONENT(a) - 25;
    if(shift > 0)
        value <<= shift;
    else if(shift < 0)
        value >>= -shift;
    if(a & SIGN_MASK)
        return -(int32_t)(value);
    return value;
}

int f16_gte(short a,short b)
{
    if(IS_ZERO(a) && IS_ZERO(b))
        return 1;
    if(IS_NAN(a) || IS_NAN(b))
        return 0;
    if((a & SIGN_MASK) == 0) {
        if((b & SIGN_MASK) == SIGN_MASK)
            return 1;
        return a >= b;
    }
    else {
        if((b & SIGN_MASK) == 0)
            return 0;
        return (a & 0x7FFF) <= (b & 0x7FFF);
    }
}

int f16_gt(short a,short b)
{
    if(IS_NAN(a) || IS_NAN(b))
        return 0;
    if(IS_ZERO(a) && IS_ZERO(b))
        return 0;
    if((a & SIGN_MASK) == 0) {
        if((b & SIGN_MASK) == SIGN_MASK)
            return 1;
        return a > b;
    }
    else {
        if((b & SIGN_MASK) == 0)
            return 0;
        return (a & 0x7FFF) < (b & 0x7FFF);
    }
    
}
int f16_eq(short a,short b)
{
    if(IS_NAN(a) || IS_NAN(b))
        return 0;
    if(IS_ZERO(a) && IS_ZERO(b))
        return 1;
    return a==b;
}

int f16_lte(short a,short b)
{
    if(IS_NAN(a) || IS_NAN(b))
        return 0;
    return f16_gte(b,a);
}

int f16_lt(short a,short b)
{
    if(IS_NAN(a) || IS_NAN(b))
        return 0;
    return f16_gt(b,a);
}

int f16_neq(short a,short b)
{
    return !f16_eq(a,b);
}

} // namespace 

float16 float16::operator + (const float16 &f)
{
  this->_value = f16_add(this->_value, f._value);
  return *this;
}

float16 float16::operator - (const float16 &f)
{
  this->_value = f16_sub(this->_value, f._value);
  return *this;
}

float16 float16::operator * (const float16 &f)
{
   this->_value = f16_mul(this->_value, f._value);
  return *this;
}

float16 float16::operator / (const float16 &f)
{
  this->_value = f16_div(this->_value, f._value);
  return *this;
}

float16& float16::operator += (const float16 &f)
{
  this->_value = f16_add(this->_value, f._value);
  return *this;
}

float16& float16::operator -= (const float16 &f)
{
   this->_value = f16_sub(this->_value, f._value);
  return *this;
}

float16& float16::operator *= (const float16 &f)
{
  this->_value = f16_mul(this->_value, f._value);
  return *this;
}

float16& float16::operator /= (const float16 &f)
{
  this->_value = f16_div(this->_value, f._value);
  return *this;
}


//////////////////////////////////////////////////////////
//
//  MATH HELPER FUNCTIONS
//
int float16::sign()
{
  if (_value & 0x8000) return -1;
  if (_value & 0xFFFF) return 1;
  return 0;
}

bool float16::isZero()
{
  return ((_value & 0x7FFF) == 0x0000);
}

bool float16::isNaN()
{
  if ((_value & 0x7C00) != 0x7C00) return false;
  if ((_value & 0x03FF) == 0x0000) return false;
  return true;
}

bool float16::isInf()
{
  return ((_value == 0x7C00) || (_value == 0xFC00));
}

bool float16::isPosInf()
{
  return (_value == 0x7C00);
}

bool float16::isNegInf()
{
  return (_value == 0xFC00);
}


//////////////////////////////////////////////////////////
//
//  CORE CONVERSION
//
float float16::f16tof32(uint16_t _value) const
{
  uint16_t sgn, man;
  int exp;
  double f;

  sgn = (_value & 0x8000) > 0;
  exp = (_value & 0x7C00) >> 10;
  man = (_value & 0x03FF);

  //  ZERO
  if ((_value & 0x7FFF) == 0)
  {
    return sgn ? -0 : 0;
  }
  //  NAN & INF
  if (exp == 0x001F)
  {
    if (man == 0) return sgn ? -INFINITY : INFINITY;
    else return NAN;
  }

  //  NORMAL
  if (exp > 0)
  {
    f = pow(2.0, exp - 15) * (1 + man * 0.0009765625);
    return sgn ? -f : f;
  }
  //  SUBNORMAL
  //  exp == 0;
  f = pow(2.0, -24) * man;
  return sgn ? -f : f;
}


uint16_t float16::f32tof16(float f) const
{
  uint32_t t = *(uint32_t *) &f;
  //  man bits = 10; but we keep 11 for rounding
  uint16_t man = (t & 0x007FFFFF) >> 12;
  int16_t  exp = (t & 0x7F800000) >> 23;
  bool     sgn = (t & 0x80000000);

// Serial.print("BEFOR:\t ");
// Serial.print(sgn, HEX);
// Serial.print(" ");
// Serial.print(man, HEX);
// Serial.print(" ");
// Serial.println(exp, HEX);

  //  handle 0
  if ((t & 0x7FFFFFFF) == 0)
  {
    return sgn ? 0x8000 : 0x0000;
  }

  //  denormalized float32 does not fit in float16
  if (exp == 0x00)
  {
    return sgn ? 0x8000 : 0x0000;
  }

  //  handle INF and NAN == infinity and not a number
  if (exp == 0x00FF)
  {
    if (man) return 0xFE00;         //  NAN
    return sgn ? 0xFC00 : 0x7C00;   //  -INF : INF
  }

  //  rescale exponent
  exp = exp - 127 + 15;

  //  overflow does not fit => INF (infinity)
  if (exp > 30)
  {
    return sgn ? 0xFC00 : 0x7C00;   //  -INF : INF
  }

  //  subnormal numbers out of range => 0.
  if (exp < -9)
  {
    return sgn ? 0x8000 : 0x0000;   //  -0 or 0  ?   just 0 ?
  }

  //  subnormal numbers
  if (exp <= 0)
  {
    exp = 0;
    man = abs(f) * 16777216;  //  pow(2.0, 24);
    if (sgn) return 0x8000 | man;
    return man;
  }


  //  normal numbers
  //  rounding
  man++;
  man >>= 1;
  //  correction mantissa overflow issue #10
  if (man == 0x0400)
  {
    exp++;
    man = 0;
  }
  exp <<= 10;


  if (sgn) return 0x8000 | exp | man;
  return exp | man;
}


//  -- END OF FILE --
