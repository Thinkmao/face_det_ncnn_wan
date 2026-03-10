#pragma once

#define LANDMARK_NUMBER 5

struct face_landmark {
    float x[LANDMARK_NUMBER];
    float y[LANDMARK_NUMBER];
};

struct face_box {
    float x0, y0, x1, y1;
    float score;
    face_landmark landmark;
};
