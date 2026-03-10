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

static inline float box_iou(const face_box& a, const face_box& b)
{
    const float inter_x0 = std::max(a.x0, b.x0);
    const float inter_y0 = std::max(a.y0, b.y0);
    const float inter_x1 = std::min(a.x1, b.x1);
    const float inter_y1 = std::min(a.y1, b.y1);
    const float iw = inter_x1 - inter_x0;
    const float ih = inter_y1 - inter_y0;
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float inter = iw * ih;
    const float area_a = std::max(0.f, a.x1 - a.x0) * std::max(0.f, a.y1 - a.y0);
    const float area_b = std::max(0.f, b.x1 - b.x0) * std::max(0.f, b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    return uni > 0.f ? (inter / uni) : 0.f;
}

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
    float score = 1.f;
    const float w = std::max(0.f, face.x1 - face.x0);
    const float h = std::max(0.f, face.y1 - face.y0);
    if (w <= 0.f || h <= 0.f) return 0.f;

    const float ar = w / h;
    const float ar_sym = ar >= 1.f ? ar : (1.f / ar);
    if (ar_sym >= params.ar_hard) {
        score *= params.ar_hard_multiplier;
    } else if (ar_sym > params.ar_start) {
        score *= std::exp(-params.ar_k * (ar_sym - params.ar_start));
    }
    return clamp01(score);
}

void apply_detection_score_suppression(std::vector<face_box>& faces, int img_w, int img_h,
                                       const ScoreSuppressionConfig& config)
{
    if (faces.empty()) return;

    std::vector<float> multiplier(faces.size(), 1.f);
    std::vector<float> pre_scores(faces.size(), 0.f);
    std::vector<float> cx(faces.size(), 0.f), cy(faces.size(), 0.f), ww(faces.size(), 0.f), hh(faces.size(), 0.f);

    for (size_t i = 0; i < faces.size(); ++i) {
        pre_scores[i] = faces[i].score;
        ww[i] = std::max(0.f, faces[i].x1 - faces[i].x0);
        hh[i] = std::max(0.f, faces[i].y1 - faces[i].y0);
        cx[i] = 0.5f * (faces[i].x0 + faces[i].x1);
        cy[i] = 0.5f * (faces[i].y0 + faces[i].y1);

        multiplier[i] *= compute_landmark_score_multiplier(faces[i], img_w, img_h, config.landmark);
    }

    // Box suppression follows original lab/main.cpp apply_fp_suppression logic:
    // size prior + aspect prior + neighborhood density prior.
    float sum_size = 0.f;
    int cnt = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        if (pre_scores[i] < config.box.prob_gate) continue;
        if (ww[i] <= 0.f || hh[i] <= 0.f) continue;
        sum_size += std::sqrt(ww[i] * hh[i]);
        cnt++;
    }
    const float avg_size = (cnt > 0) ? (sum_size / cnt) : 0.f;
    const float size_thr = avg_size * config.box.size_ratio_thr;

    for (size_t i = 0; i < faces.size(); ++i) {
        if (pre_scores[i] < config.box.prob_gate) continue;

        float box_m = 1.f;
        const float w = ww[i], h = hh[i];
        if (w <= 0.f || h <= 0.f) {
            multiplier[i] = 0.f;
            continue;
        }

        if (size_thr > 0.f) {
            const float s = std::sqrt(w * h);
            if (s > 0.f && s < size_thr) {
                box_m *= clamp01(s / size_thr);
            }
        }

        const float ar = w / h;
        const float ar_sym = ar >= 1.f ? ar : (1.f / ar);
        if (ar_sym >= config.box.ar_hard) {
            box_m *= config.box.ar_hard_multiplier;
        } else if (ar_sym > config.box.ar_start) {
            box_m *= std::exp(-config.box.ar_k * (ar_sym - config.box.ar_start));
        }

        const float r = config.box.density_r_scale * std::min(w, h) + config.box.density_bias_px;
        const float r2 = r * r;
        int nbr = 0;
        for (size_t j = 0; j < faces.size(); ++j) {
            if (i == j) continue;
            if (pre_scores[j] < config.box.prob_gate) continue;

            const float dx = cx[j] - cx[i];
            const float dy = cy[j] - cy[i];
            if (dx * dx + dy * dy > r2) continue;
            if (box_iou(faces[i], faces[j]) < config.box.density_iou_thr) continue;

            nbr++;
            if (nbr >= config.box.density_min_nbr) break;
        }
        if (nbr < config.box.density_min_nbr) {
            box_m *= (nbr == 0) ? config.box.isolated_zero_nbr_multiplier : config.box.isolated_one_nbr_multiplier;
        }

        multiplier[i] *= clamp01(box_m);
    }

    for (size_t i = 0; i < faces.size(); ++i) {
        faces[i].score = clamp01(faces[i].score * multiplier[i]);
    }
}
