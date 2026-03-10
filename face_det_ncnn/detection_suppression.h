#pragma once

#include <vector>
#include "face_types.h"

struct LandmarkSuppressionParams {
    float margin_ratio = 0.2f;
    int min_inside_landmarks = 4;

    float eye_mouth_min_ratio = 0.02f;
    float eye_mouth_max_ratio = 0.95f;

    float eye_mouth_vertical_min_ratio = 0.08f;
    float eye_mouth_vertical_max_ratio = 1.05f;

    float nose_dist_min_diag_ratio = 0.02f;
    float nose_dist_max_diag_ratio = 0.85f;

    float y_slack_ratio = 0.35f;

    float lmk_span_y_min_ratio = 0.08f;
    float lmk_span_y_max_ratio = 1.1f;

    float centroid_min_ratio = -0.35f;
    float centroid_max_ratio = 1.35f;

    float out_of_image_margin_ratio = 0.3f;

    float fail_multiplier = 0.65f;
};

struct BoxSuppressionParams {
    float min_face_size = 10.f;
    float max_aspect_ratio = 1.8f;
    float min_aspect_ratio = 0.55f;

    float size_fail_multiplier = 0.75f;
    float aspect_fail_multiplier = 0.8f;
};

struct IsolationSuppressionParams {
    float isolated_dist_ratio = 2.8f;
    float isolated_fail_multiplier = 0.85f;
};

struct ScoreSuppressionConfig {
    float draw_threshold = 0.6f;
    LandmarkSuppressionParams landmark;
    BoxSuppressionParams box;
    IsolationSuppressionParams isolation;
};

float compute_landmark_score_multiplier(const face_box& face, int img_w, int img_h,
                                        const LandmarkSuppressionParams& params);

float compute_box_score_multiplier(const face_box& face, int img_w, int img_h,
                                   const BoxSuppressionParams& params);

void apply_detection_score_suppression(std::vector<face_box>& faces, int img_w, int img_h,
                                       const ScoreSuppressionConfig& config);
