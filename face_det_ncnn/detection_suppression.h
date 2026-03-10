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
    // Original lab/main.cpp apply_fp_suppression style params
    float prob_gate = 0.50f;
    float size_ratio_thr = 0.60f;

    float ar_start = 1.80f;
    float ar_hard = 3.00f;
    float ar_k = 1.20f;

    float density_iou_thr = 0.05f;
    int density_min_nbr = 2;
    float density_r_scale = 0.25f;
    float density_bias_px = 4.0f;

    // Penalty factors from original logic
    float ar_hard_multiplier = 0.05f;
    float isolated_zero_nbr_multiplier = 0.35f;
    float isolated_one_nbr_multiplier = 0.65f;
};

struct ScoreSuppressionConfig {
    float draw_threshold = 0.6f;
    LandmarkSuppressionParams landmark;
    BoxSuppressionParams box;
};

float compute_landmark_score_multiplier(const face_box& face, int img_w, int img_h,
                                        const LandmarkSuppressionParams& params);

float compute_box_score_multiplier(const face_box& face, int img_w, int img_h,
                                   const BoxSuppressionParams& params);

void apply_detection_score_suppression(std::vector<face_box>& faces, int img_w, int img_h,
                                       const ScoreSuppressionConfig& config);
