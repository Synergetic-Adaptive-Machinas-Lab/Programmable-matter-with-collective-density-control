#include "command.h"
#include <string.h>
#include <math.h>

void readExactBytes(Stream& s, uint8_t* dst, int n) {
  int got = 0;
  while (got < n) {
    int avail = s.available();
    if (avail <= 0) continue;

    int take = avail;
    if (take > (n - got)) take = (n - got);

    int r = s.readBytes((char*)(dst + got), take);
    got += r;
  }
}

void writeExactBytes(Stream& s, const uint8_t* src, int n) {
  int sent = 0;
  while (sent < n) {
    int w = s.write(src + sent, n - sent);
    sent += w;
  }
}

uint16_t rd_u16_le(const uint8_t* p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

uint32_t rd_u32_le(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void wr_u16_le(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
}

void wr_u32_le(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF);
  p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static inline uint16_t crc16_update(uint16_t crc, uint8_t data) {
  crc ^= ((uint16_t)data << 8);
  for (int i = 0; i < 8; i++) {
    crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

uint16_t crc16_ccitt(const uint8_t* data, int n, uint16_t init) {
  uint16_t crc = init;
  for (int i = 0; i < n; i++) crc = crc16_update(crc, data[i]);
  return crc;
}

void buildX(const uint8_t* packed256, uint8_t* X512) {
  for (int i = 0; i < 256; i++) {
    uint8_t b = packed256[i];
    X512[2*i + 0] = (uint8_t)(b & 0x0F);
    X512[2*i + 1] = (uint8_t)((b >> 4) & 0x0F);
  }
}

static constexpr uint8_t  BASE_ADDR  = 0x40;
static constexpr int      NUM_BOARDS = 32;
static constexpr int      MAG_PER_BRD = 8;
static constexpr uint16_t PWM_MAX    = 4095;

static inline uint16_t intensityToPwm(int intensity, float gamma) {
  if (intensity == 0) return 0;
  int mag = abs(intensity);
  float t = (float)mag / 7.0f;

  float frac;
  if (fabsf(gamma) < 1e-6f) {
    frac = t;
  } else {

    frac = (1.0f - expf(-gamma * t)) / (1.0f - expf(-gamma));
  }

  uint16_t pwm = (uint16_t)(frac * (float)PWM_MAX + 0.5f);
  return (pwm > PWM_MAX) ? PWM_MAX : pwm;
}

static bool writeBoardAllChannels(TwoWire& wire, uint8_t addr, const uint16_t* off) {
  wire.beginTransmission(addr);
  wire.write(0x06);
  for (int ch = 0; ch < 16; ++ch) {
    uint16_t pwm = off[ch] & 0x0FFF;

    wire.write((uint8_t)0x00);
    wire.write((uint8_t)0x00);

    wire.write((uint8_t)(pwm & 0xFF));
    wire.write((uint8_t)((pwm >> 8) & 0x0F));
  }
  int rc = wire.endTransmission(true);
  (void)rc;
  return (rc == 0);
}

void actionX(Adafruit_PWMServoDriver* boards0[NUM_BOARDS], Adafruit_PWMServoDriver* boards1[NUM_BOARDS], const uint8_t* X512, int8_t gamma) {
  const float gammaF = (float)gamma;

  auto applyBus = [&](Adafruit_PWMServoDriver* boards[NUM_BOARDS], const uint8_t* Xbase, TwoWire& wire) {
    for (int dev = 0; dev < NUM_BOARDS; ++dev) {
      Adafruit_PWMServoDriver* b = boards[dev];
      if (!b) continue;

      uint16_t off[16];
      for (int ch = 0; ch < 16; ++ch) off[ch] = 0;

      for (int m = 0; m < MAG_PER_BRD; ++m) {
        const uint8_t value = Xbase[dev * MAG_PER_BRD + m];

        if (value == 15) {
              continue;
        }

        const int intensity = (int)value - 7;
        const uint16_t pwm  = intensityToPwm(intensity, gammaF);

        const int left  = 2 * m;
        const int right = left + 1;

        if (intensity > 0) {
          off[left]  = 4095;
          off[right] = 4095 - pwm;
        } else if (intensity < 0) {
          off[left]  = 4095 - pwm;
          off[right] = 4095;
        } else {
          off[left]  = 0;
          off[right] = 0;
        }
      }

      uint8_t addr = (uint8_t)(BASE_ADDR + dev);
      (void)writeBoardAllChannels(wire, addr, off);
    }
  };

  applyBus(boards0, X512, Wire);

  applyBus(boards1, X512 + 256, Wire1);
}

void makeAck(uint8_t* ack7, uint32_t seq, uint8_t status) {
  wr_u16_le(&ack7[0], ACK_MAGIC);
  wr_u32_le(&ack7[2], seq);
  ack7[6] = status;
}

bool readAck(Stream& s, uint32_t expected_seq, uint8_t* out_status, uint32_t timeout_us) {
  uint8_t buf[ACK_BYTES];
  int idx = 0;
  uint32_t t0 = micros();

  while ((micros() - t0) < timeout_us) {
    while (s.available() && idx < ACK_BYTES) {
      buf[idx++] = (uint8_t)s.read();
    }
    if (idx < ACK_BYTES) continue;

    if (rd_u16_le(&buf[0]) == ACK_MAGIC && rd_u32_le(&buf[2]) == expected_seq) {
      if (out_status) *out_status = buf[6];
      return true;
    }

    memmove(buf, buf + 1, ACK_BYTES - 1);
    idx = ACK_BYTES - 1;
  }
  return false;
}
