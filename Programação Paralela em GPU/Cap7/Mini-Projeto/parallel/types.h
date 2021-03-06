#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

struct rect_t{
	uint8_t x;
	uint8_t y;
	uint8_t w;
	uint8_t h;
	int8_t weight;
};

struct rectBig_t{
	int x;
	int y;
	int w;
	int h;
};

struct feature_t{
	rect_t black;
	rect_t white;
	rect_t third;
};

struct stage_t{
	feature_t feature;
	float threshold;
	float leftWeight;
	float rightWeight;
};

struct stageMeta_t{
	uint16_t size;
	uint16_t start;
	float threshold;
};

#endif TYPES_H