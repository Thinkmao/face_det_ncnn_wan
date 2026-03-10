#include "detection_suppression.h"

#include <algorithm>
#include <cmath>

namespace {

static inline float clamp01(float v)
{
    return std::max(0.f, std::min(1.f, v));
}

static inline bool point_inside_margin_box(float x, float y, const face_box& box, float margin_ratio)
{
    const float bw = std::max(1.0f, box.x1 - box.x0);
    const float bh = std::max(1.0f, box.y1 - box.y0);
    const float mx = bw * margin_ratio;
    const float my = bh * margin_ratio;
    return x >= (box.x0 - mx) && x <= (box.x1 + mx) && y >= (box.y0 - my) && y <= (box.y1 + my);
}

static inline float bbox_width(const face_box& b) { return std::max(1.0f, b.x1 - b.x0); }
static inline float bbox_height(const face_box& b) { return std::max(1.0f, b.y1 - b.y0); }

} // namespace

float compute_landmark_score_multiplier(const face_box& face, int img_w, int img_h,
                                        const LandmarkSuppressionParams& params)
{
    float m = 1.f;

    const float bw = bbox_width(face);
    const float bh = bbox_height(face);

    int inside_cnt = 0;
    for (int i = 0; i < LANDMARK_NUMBER; ++i) {
        if (point_inside_margin_box(face.landmark.x[i], face.landmark.y[i], face, params.margin_ratio)) {
            inside_cnt++;
        }
    }
    if (inside_cnt < params.min_inside_landmarks) m *= params.fail_multiplier;

    const float eye_dx = face.landmark.x[1] - face.landmark.x[0];
    const float eye_dy = face.landmark.y[1] - face.landmark.y[0];
    const float eye_dist = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy);
    const float mouth_dx = face.landmark.x[4] - face.landmark.x[3];
    const float mouth_dy = face.landmark.y[4] - face.landmark.y[3];
    const float mouth_dist = std::sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy);

    const float nose_x = face.landmark.x[2];
    const float eye_min_x = std::min(face.landmark.x[0], face.landmark.x[1]);
    const float eye_max_x = std::max(face.landmark.x[0], face.landmark.x[1]);
    const float mouth_min_x = std::min(face.landmark.x[3], face.landmark.x[4]);
    const float mouth_max_x = std::max(face.landmark.x[3], face.landmark.x[4]);
    const bool nose_on_same_side_of_eyes_and_mouth =
        (nose_x < eye_min_x && nose_x < mouth_min_x) ||
        (nose_x > eye_max_x && nose_x > mouth_max_x);

    const float eye_ratio = eye_dist / bw;
    const float mouth_ratio = mouth_dist / bw;

    if (!nose_on_same_side_of_eyes_and_mouth && eye_ratio < params.eye_mouth_min_ratio) m *= params.fail_multiplier;
    if (!nose_on_same_side_of_eyes_and_mouth && mouth_ratio < params.eye_mouth_min_ratio) m *= params.fail_multiplier;
    if (eye_ratio > params.eye_mouth_max_ratio) m *= params.fail_multiplier;
    if (mouth_ratio > params.eye_mouth_max_ratio) m *= params.fail_multiplier;

    const float eye_cx = 0.5f * (face.landmark.x[0] + face.landmark.x[1]);
    const float eye_cy = 0.5f * (face.landmark.y[0] + face.landmark.y[1]);
    const float mouth_cx = 0.5f * (face.landmark.x[3] + face.landmark.x[4]);
    const float mouth_cy = 0.5f * (face.landmark.y[3] + face.landmark.y[4]);
    const float nose_y = face.landmark.y[2];

    const float eye_mouth_dy_ratio = std::fabs(mouth_cy - eye_cy) / bh;
    if (eye_mouth_dy_ratio < params.eye_mouth_vertical_min_ratio || eye_mouth_dy_ratio > params.eye_mouth_vertical_max_ratio) {
        m *= params.fail_multiplier;
    }

    const float nose_eye_dist = std::sqrt((nose_x - eye_cx) * (nose_x - eye_cx) +
                                          (nose_y - eye_cy) * (nose_y - eye_cy));
    const float nose_mouth_dist = std::sqrt((nose_x - mouth_cx) * (nose_x - mouth_cx) +
                                            (nose_y - mouth_cy) * (nose_y - mouth_cy));
    const float box_diag = std::sqrt(bw * bw + bh * bh);
    const float nose_eye_ratio = nose_eye_dist / box_diag;
    const float nose_mouth_ratio = nose_mouth_dist / box_diag;
    if (nose_eye_ratio < params.nose_dist_min_diag_ratio || nose_eye_ratio > params.nose_dist_max_diag_ratio) {
        m *= params.fail_multiplier;
    }
    if (nose_mouth_ratio < params.nose_dist_min_diag_ratio || nose_mouth_ratio > params.nose_dist_max_diag_ratio) {
        m *= params.fail_multiplier;
    }

    const float y_slack = params.y_slack_ratio * bh;
    if (mouth_cy < eye_cy - y_slack) m *= params.fail_multiplier;
    if (nose_y < eye_cy - y_slack || nose_y > mouth_cy + y_slack) m *= params.fail_multiplier;

    float min_ly = face.landmark.y[0], max_ly = face.landmark.y[0];
    for (int i = 1; i < LANDMARK_NUMBER; ++i) {
        min_ly = std::min(min_ly, face.landmark.y[i]);
        max_ly = std::max(max_ly, face.landmark.y[i]);
    }
    const float lmk_span_y_ratio = (max_ly - min_ly) / bh;
    if (lmk_span_y_ratio < params.lmk_span_y_min_ratio || lmk_span_y_ratio > params.lmk_span_y_max_ratio) {
        m *= params.fail_multiplier;
    }

    const float cx = 0.2f * (face.landmark.x[0] + face.landmark.x[1] + face.landmark.x[2] + face.landmark.x[3] + face.landmark.x[4]);
    const float cy = 0.2f * (face.landmark.y[0] + face.landmark.y[1] + face.landmark.y[2] + face.landmark.y[3] + face.landmark.y[4]);
    const float cx_ratio = (cx - face.x0) / bw;
    const float cy_ratio = (cy - face.y0) / bh;
    if (cx_ratio < params.centroid_min_ratio || cx_ratio > params.centroid_max_ratio ||
        cy_ratio < params.centroid_min_ratio || cy_ratio > params.centroid_max_ratio) {
        m *= params.fail_multiplier;
    }

    const float out_margin = params.out_of_image_margin_ratio;
    if (face.x1 < -out_margin * bw || face.y1 < -out_margin * bh ||
        face.x0 > img_w + out_margin * bw || face.y0 > img_h + out_margin * bh) {
        m *= params.fail_multiplier;
    }

    return clamp01(m);
}

float compute_box_score_multiplier(const face_box& face, int img_w, int img_h,
                                   const BoxSuppressionParams& params)
{
    (void)img_w;
    (void)img_h;
    float m = 1.f;
    const float bw = bbox_width(face);
    const float bh = bbox_height(face);
    if (std::min(bw, bh) < params.min_face_size) m *= params.size_fail_multiplier;

    const float ar = bw / bh;
    if (ar < params.min_aspect_ratio || ar > params.max_aspect_ratio) m *= params.aspect_fail_multiplier;
    return clamp01(m);
}

void apply_detection_score_suppression(std::vector<face_box>& faces, int img_w, int img_h,
                                       const ScoreSuppressionConfig& config)
{
    if (faces.empty()) return;

    std::vector<float> base_multiplier(faces.size(), 1.f);
    for (size_t i = 0; i < faces.size(); ++i) {
        float m = 1.f;
        m *= compute_landmark_score_multiplier(faces[i], img_w, img_h, config.landmark);
        m *= compute_box_score_multiplier(faces[i], img_w, img_h, config.box);
        base_multiplier[i] = clamp01(m);
    }

    if (faces.size() > 1) {
        for (size_t i = 0; i < faces.size(); ++i) {
            const float cxi = 0.5f * (faces[i].x0 + faces[i].x1);
            const float cyi = 0.5f * (faces[i].y0 + faces[i].y1);
            float min_center_dist = 1e12f;
            for (size_t j = 0; j < faces.size(); ++j) {
                if (i == j) continue;
                const float cxj = 0.5f * (faces[j].x0 + faces[j].x1);
                const float cyj = 0.5f * (faces[j].y0 + faces[j].y1);
                const float dx = cxi - cxj;
                const float dy = cyi - cyj;
                const float d = std::sqrt(dx * dx + dy * dy);
                min_center_dist = std::min(min_center_dist, d);
            }
            const float diag = std::sqrt(bbox_width(faces[i]) * bbox_width(faces[i]) +
                                         bbox_height(faces[i]) * bbox_height(faces[i]));
            if (diag > 1e-6f && min_center_dist / diag > config.isolation.isolated_dist_ratio) {
                base_multiplier[i] *= config.isolation.isolated_fail_multiplier;
            }
        }
    }

    for (size_t i = 0; i < faces.size(); ++i) {
        faces[i].score = clamp01(faces[i].score * base_multiplier[i]);
    }
}
