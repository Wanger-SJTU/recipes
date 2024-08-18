

#include "float16.h"
#include <iostream>
#include <vector>
#include <bitset> //输出二进制的头文件

using namespace std;

void NVIDIA_fast_dequant_u8() 
{
    vector<int8_t> quant_int8 = {1, 3, 4, 24};
    
    for (auto item : quant_int8) {
        uint16_t new_value = (0x6400 | item);
        cout << "origin " << static_cast<int>(item) << " float " << float16(new_value).toFloat() - 1024 << endl;
        // cout << "origin " <<  bitset<sizeof(item)*8>(item) << " float " << hex << float16(new_value).getBinary()  << endl;
    }
}

void NVIDIA_fast_dequant_s8() 
{
    vector<int8_t> quant_int8 = {-1, -3, -4, -24};
    
    for (auto item : quant_int8) {
        uint16_t sign =  (item & 0x0080) << 8;
        int8_t trueForm = (item & 0x80) ? (item & 0x80) | (~item + 1) : item;
        uint16_t new_value = (0x6400 | (item & 0x7F));
        float16 base = float16(uint16_t(0x6480));
        cout << "origin " << static_cast<int>(item) << " float " << (float16(new_value) - base).toFloat() << endl;
        // cout << "origin " <<  bitset<sizeof(item)*8>(item) << " float " << hex << float16(new_value).getBinary()  << endl;
    }
}

void new_fast_dequant() 
{
    vector<int8_t> quant_int8 = {-1, 3, 4, 24};
    
    for (auto item : quant_int8) {
        uint16_t new_value1 = (0x007F & (1 << 7 | 1)) << 3;
        uint16_t new_value2 = (0x0080 & (1 << 7 | 1)) << 3;
        // cout << " origin " <<  bitset<sizeof(item)*8>(1 << 7 | 1)  << 
        //         " float1 " <<  bitset<sizeof(uint16_t)*8>(float16(new_value1).getBinary())  << 
        //         " float2 " <<  bitset<sizeof(uint16_t)*8>(float16(new_value2).getBinary())  << endl;

        cout << "origin " << static_cast<int>(item) << 
              " float " << float16(new_value1).toFloat() - float16(new_value2).toFloat() << endl;
    }
}

int main() 
{
    // NVIDIA_fast_dequant();
    NVIDIA_fast_dequant_s8();
    return 0;
}