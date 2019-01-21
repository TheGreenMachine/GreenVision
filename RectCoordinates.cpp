#include "RectCoordinates.h"

RectCoordinates::RectCoordinates(float topLeftX, float topLeftY, float width, float height, float bottomRightX,
                                 float bottomRightY, int centerX, int centerY) : topLeftX(topLeftX), topLeftY(topLeftY),
                                                                                 width(width), height(height),
                                                                                 bottomRightX(bottomRightX),
                                                                                 bottomRightY(bottomRightY),
                                                                                 centerX(centerX), centerY(centerY) {}

float RectCoordinates::getTopLeftX() const {
    return topLeftX;
}

float RectCoordinates::getTopLeftY() const {
    return topLeftY;
}

float RectCoordinates::getWidth() const {
    return width;
}

float RectCoordinates::getHeight() const {
    return height;
}

float RectCoordinates::getBottomRightX() const {
    return bottomRightX;
}

float RectCoordinates::getBottomRightY() const {
    return bottomRightY;
}

int RectCoordinates::getCenterX() const {
    return centerX;
}

int RectCoordinates::getCenterY() const {
    return centerY;
}
