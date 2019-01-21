/**
 * @author Luke L
 * Class that holds the coordinates from the OpenCV rectangles
 */

#ifndef GREENVISION_RECTCOORDINATES_H
#define GREENVISION_RECTCOORDINATES_H


class RectCoordinates {
private:
    float topLeftX;
    float topLeftY;
    float width;
    float height;
    float bottomRightX;
    float bottomRightY;
    int centerX;
    int centerY;

public:
    RectCoordinates(float topLeftX, float topLeftY, float width, float height, float bottomRightX, float bottomRightY,
                    int centerX, int centerY);

    float getTopLeftX() const;

    float getTopLeftY() const;

    float getWidth() const;

    float getHeight() const;

    float getBottomRightX() const;

    float getBottomRightY() const;

    int getCenterX() const;

    int getCenterY() const;
};


#endif //GREENVISION_RECTCOORDINATES_H
