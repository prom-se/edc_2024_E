#ifndef SERIAL_TEST_PACKET_H
#define SERIAL_TEST_PACKET_H

#include <cstdint>

#pragma pack(2)
struct VisionMsg{
    uint16_t head;
    float chess_x;
    float chess_y;
    float dst_x;
    float dst_y;
};

union VisionPack{
    struct VisionMsg msg;
    uint8_t bytes[sizeof(VisionMsg)];
};
#pragma pack()

#endif //SERIAL_TEST_PACKET_H