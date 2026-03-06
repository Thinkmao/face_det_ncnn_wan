#include <opencv2/opencv.hpp>
#include "net.h"
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <cstring>
#include <cmath>
//#include <time.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

using namespace std;

// ====================== ������� + 5����� ======================
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

static void qsort_descent_inplace(face_box* faceobjects, int left, int right) {
	int i = left, j = right;
	float p = faceobjects[(left + right) / 2].score;
	while (i <= j) {
		while (faceobjects[i].score > p) i++;
		while (faceobjects[j].score < p) j--;
		if (i <= j) {
			face_box a = faceobjects[i], b = faceobjects[j];
			faceobjects[i] = b; faceobjects[j] = a;
			i++; j--;
		}
	}
	if (left < j) qsort_descent_inplace(faceobjects, left, j);
	if (i < right) qsort_descent_inplace(faceobjects, i, right);
}
static void qsort_descent_inplace(face_box* faceObjects, int faceCount) {
	if (faceCount > 0) qsort_descent_inplace(faceObjects, 0, faceCount - 1);
}
static inline float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
	float xmin1, float ymin1, float xmax1, float ymax1)
{
	float w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
	float h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
	float i = w * h;
	float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
		(xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
	return u <= 0.f ? 0.f : (i / u);
}
static int nms_sorted_bboxes(face_box* faceobjects, const int faceCount, int* picked, const float nms_threshold)
{
	int n = 0;
	for (int i = 0; i < faceCount; i++) {
		const face_box a = faceobjects[i];
		int keep = 1;
		for (int j = 0; j < n; j++) {
			const face_box b = faceobjects[picked[j]];
			if (CalculateOverlap(a.x0, a.y0, a.x1, a.y1, b.x0, b.y0, b.x1, b.y1) > nms_threshold)
				keep = 0;
		}
		if (keep) picked[n++] = i;
	}
	return n;
}

static int generate_anchors(int step, const int* min_sizes,
	float* score_blob, float* bbox_blob, float* landmark_blob,
	int box_w, int box_h, int score_w, int score_h, int land_w, int land_h,
	face_box* faceobjects, const int* target_size, const int* img_size)
{
	int feature_map_w = target_size[0] / step;
	int feature_map_h = target_size[1] / step;
	int faceCount = 0;
	for (int i = 0; i < feature_map_w; i++) {
		for (int j = 0; j < feature_map_h; j++) {
			for (int min_size_index = 0; min_size_index < 2; min_size_index++) {
				int channel_index = i * feature_map_w * 2 + j * 2 + min_size_index;

				float s_kx = (float)min_sizes[min_size_index] / (float)target_size[0];
				float s_ky = (float)min_sizes[min_size_index] / (float)target_size[1];
				float dense_cx = (j + 0.5f) * step / target_size[0];
				float dense_cy = (i + 0.5f) * step / target_size[1];

				int index = (channel_index % 2 == 1) ? ((channel_index - 1) / 2) : (channel_index / 2);
				int scale = channel_index % 2;

				float prob = score_blob[score_w * score_h * scale * 2 + index];
				float prob_ = score_blob[score_w * score_h * (scale * 2 + 1) + index];

				float sum_prob = (std::exp(prob) + std::exp(prob_));
				prob_ = sum_prob != 0.f ? std::exp(prob_) / sum_prob : 0.f;

				if (prob_ > 0.5f) {
					float dx = bbox_blob[box_w * box_h * scale * 4 + index];
					float dy = bbox_blob[box_w * box_h * (scale * 4 + 1) + index];
					float dw = bbox_blob[box_w * box_h * (scale * 4 + 2) + index];
					float dh = bbox_blob[box_w * box_h * (scale * 4 + 3) + index];

					float bbox_x = dense_cx + dx * 0.1f * s_kx;
					float bbox_y = dense_cy + dy * 0.1f * s_ky;
					float bbox_w = s_kx * std::exp(dw * 0.2f);
					float bbox_h = s_ky * std::exp(dh * 0.2f);

					bbox_x -= (bbox_w / 2);
					bbox_y -= (bbox_h / 2);
					bbox_w += bbox_x;
					bbox_h += bbox_y;

					face_box obj;
					obj.x0 = bbox_x * target_size[0];
					obj.y0 = bbox_y * target_size[1];
					obj.x1 = bbox_w * target_size[0];
					obj.y1 = bbox_h * target_size[1];

					for (int landmark_idx = 0; landmark_idx < LANDMARK_NUMBER * 2; landmark_idx++) {
						if (landmark_idx % 2 == 0) {
							obj.landmark.x[landmark_idx / 2] =
								(dense_cx + landmark_blob[land_w * land_h * (scale * 5 * 2 + landmark_idx) + index] * 0.1f * s_kx) * target_size[0];
						}
						else {
							obj.landmark.y[landmark_idx / 2] =
								(dense_cy + landmark_blob[land_w * land_h * (scale * 5 * 2 + landmark_idx) + index] * 0.1f * s_ky) * target_size[1];
						}
					}
					obj.score = prob_;
					faceobjects[faceCount++] = obj;
				}
			}
		}
	}
	(void)img_size; // δʹ�ã�����������ǩ��
	return faceCount;
}

static int detect_retinaface_forward(const ncnn::Mat& bgr, ncnn::Mat& output, std::vector<face_box>& out_p_faces)
{
	// ��ʼ������ָ��Ϊnullptr
	face_box* box_list_32 = nullptr;
	face_box* box_list_16 = nullptr;
	face_box* box_list_8 = nullptr;
	face_box* total_face_box = nullptr;
	int* picked = nullptr;
	face_box* detectResult = nullptr;

	int ret = 0; // Ĭ�Ϸ���ֵ

	const float prob_threshold = 0.35f;
	const float nms_threshold = 0.3f;

	int img_w = bgr.w;
	int img_h = bgr.h;

	int target_size_[2] = { 320, 320 };
	const int img_size[2] = { img_w, img_h };

	// �������ű���
	float scale = std::min(static_cast<float>(target_size_[0]) / img_w, static_cast<float>(target_size_[1]) / img_h);
	// �������ź�ĳߴ�
	int new_w = static_cast<int>(img_w * scale);
	int new_h = static_cast<int>(img_h * scale);
	int top = (target_size_[1] - new_h) / 2;
	int left = (target_size_[0] - new_w) / 2;

	int ptr_ = 0;

	/****************************** Stride 32 ���� ******************************/
	// ֱ��ʹ��output.data�е�ָ�룬����malloc
	float* bbox_blob_32 = &((float*)output.data)[ptr_];
	ptr_ += 8 * 40 * 40;

	float* score_blob_32 = &((float*)output.data)[ptr_];
	ptr_ += 4 * 40 * 40;

	float* landmark_blob_32 = &((float*)output.data)[ptr_];
	ptr_ += 20 * 40 * 40;

	int step_32 = 8;
	int min_sizes_32[2] = { 16, 32 };

	// ����box_list�ڴ�
	box_list_32 = (face_box*)malloc(sizeof(face_box) * 40 * 40 * 2);
	if (!box_list_32)
	{
		ret = -3002;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	int faceCount32 = generate_anchors(step_32, min_sizes_32, score_blob_32, bbox_blob_32, landmark_blob_32,
		40, 40, 40, 40, 40, 40, box_list_32, target_size_, img_size);

	/****************************** Stride 16 ���� ******************************/
	float* bbox_blob_16 = &((float*)output.data)[ptr_];
	ptr_ += 8 * 20 * 20;

	float* score_blob_16 = &((float*)output.data)[ptr_];
	ptr_ += 4 * 20 * 20;

	float* landmark_blob_16 = &((float*)output.data)[ptr_];
	ptr_ += 20 * 20 * 20;

	int step16 = 16;
	int min_sizes16[2] = { 64, 128 };

	box_list_16 = (face_box*)malloc(sizeof(face_box) * 20 * 20 * 2);
	if (!box_list_16)
	{
		ret = -3003;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	int faceCount16 = generate_anchors(step16, min_sizes16, score_blob_16, bbox_blob_16, landmark_blob_16,
		20, 20, 20, 20, 20, 20, box_list_16, target_size_, img_size);

	/****************************** Stride 8 ���� ******************************/
	float* bbox_blob_8 = &((float*)output.data)[ptr_];
	ptr_ += 8 * 10 * 10;

	float* score_blob_8 = &((float*)output.data)[ptr_];
	ptr_ += 4 * 10 * 10;

	float* landmark_blob_8 = &((float*)output.data)[ptr_];
	// ע�⣺���ﲻ��Ҫ���ƶ�ptr����Ϊ�������һ��blob

	int step8 = 32;
	int min_sizes8[2] = { 256, 512 };

	box_list_8 = (face_box*)malloc(sizeof(face_box) * 10 * 10 * 2);
	if (!box_list_8)
	{
		ret = -3004;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	int faceCount8 = generate_anchors(step8, min_sizes8, score_blob_8, bbox_blob_8, landmark_blob_8,
		10, 10, 10, 10, 10, 10, box_list_8, target_size_, img_size);

	/****************************** �ϲ���� ******************************/
	int total_face_count = faceCount32 + faceCount16 + faceCount8;
	if (total_face_count == 0)
	{
		ret = -3001;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	total_face_box = (face_box*)malloc(sizeof(face_box) * total_face_count);
	if (!total_face_box)
	{
		ret = -3005;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	// �޸�ָ��ƫ�ƣ�ʹ��Ԫ��ƫ�ƶ����ֽ�ƫ��
	memcpy(total_face_box, box_list_32, sizeof(face_box) * faceCount32);
	memcpy(total_face_box + faceCount32, box_list_16, sizeof(face_box) * faceCount16);
	memcpy(total_face_box + faceCount32 + faceCount16, box_list_8, sizeof(face_box) * faceCount8);

	// �����Ŷ�����
	qsort_descent_inplace(total_face_box, total_face_count);

	/****************************** NMS���� ******************************/
	picked = (int*)malloc(sizeof(int) * total_face_count);
	if (!picked)
	{
		ret = -3006;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	int pickedFaceCount = nms_sorted_bboxes(total_face_box, total_face_count, picked, nms_threshold);

	/****************************** �������ս�� ******************************/
	detectResult = (face_box*)malloc(sizeof(face_box) * pickedFaceCount);
	if (!detectResult)
	{
		ret = -3007;
		if (box_list_32)
			free(box_list_32);
		if (box_list_16)
			free(box_list_16);
		if (box_list_8)
			free(box_list_8);

		// �ͷ��м���
		if (total_face_box)
			free(total_face_box);
		if (picked)
			free(picked);
		if (detectResult)
			free(detectResult);
		return ret;
	}

	for (int i = 0; i < pickedFaceCount; i++)
	{
		// ���ƽṹ��
		detectResult[i] = total_face_box[picked[i]];

		// padding������ת������320x320�ռ�ת����ԭʼͼ��ռ�
		detectResult[i].x0 = (detectResult[i].x0 - left) / scale;
		detectResult[i].y0 = (detectResult[i].y0 - top) / scale;
		detectResult[i].x1 = (detectResult[i].x1 - left) / scale;
		detectResult[i].y1 = (detectResult[i].y1 - top) / scale;
		// �ؼ�������ת��
		for (int landmark_idx = 0; landmark_idx < LANDMARK_NUMBER; landmark_idx++) {
			detectResult[i].landmark.x[landmark_idx] = (detectResult[i].landmark.x[landmark_idx] - left) / scale;
			detectResult[i].landmark.y[landmark_idx] = (detectResult[i].landmark.y[landmark_idx] - top) / scale;
		}

		// ���ӵ��������
		out_p_faces.push_back(detectResult[i]);
	}

	if (box_list_32)
		free(box_list_32);
	if (box_list_16)
		free(box_list_16);
	if (box_list_8)
		free(box_list_8);

	// �ͷ��м���
	if (total_face_box)
		free(total_face_box);
	if (picked)
		free(picked);
	if (detectResult)
		free(detectResult);
	return ret;
}

//static int detect_retinaface_forward(ncnn::Mat& bgr, ncnn::Mat& output, std::vector<face_box>& out_p_faces)
//{
//    const float nms_threshold = 0.3f;
//
//    const int img_w = bgr.w;
//    const int img_h = bgr.h;
//
//    const int target_size_[2] = { 320, 320 };
//    const int img_size_arr[2] = { img_w, img_h };
//
//    // ����Ԥ����ʱ�����ź�������
//    float scale = std::min(static_cast<float>(target_size_[0]) / img_w,
//        static_cast<float>(target_size_[1]) / img_h);
//    int new_w = static_cast<int>(img_w * scale);
//    int new_h = static_cast<int>(img_h * scale);
//    int top = (target_size_[1] - new_h) / 2;
//    int left = (target_size_[0] - new_w) / 2;
//
//    int ptr_ = 0;
//
//    // -------- stride 32 (feature 40x40) --------
//    float* bbox_blob_32 = &((float*)output.data)[ptr_]; ptr_ += 8 * 40 * 40;
//    float* score_blob_32 = &((float*)output.data)[ptr_]; ptr_ += 4 * 40 * 40;
//    float* landmark_blob_32 = &((float*)output.data)[ptr_]; ptr_ += 20 * 40 * 40;
//
//    int min_sizes_32[2] = { 16, 32 };
//    std::vector<face_box> box_list_32(40 * 40 * 2);
//    const int faceCount32 = generate_anchors(8, min_sizes_32,
//        score_blob_32, bbox_blob_32, landmark_blob_32,
//        40, 40, 40, 40, 40, 40,
//        box_list_32.data(), target_size_, img_size_arr);
//
//    // -------- stride 16 (feature 20x20) --------
//    float* bbox_blob_16 = &((float*)output.data)[ptr_]; ptr_ += 8 * 20 * 20;
//    float* score_blob_16 = &((float*)output.data)[ptr_]; ptr_ += 4 * 20 * 20;
//    float* landmark_blob_16 = &((float*)output.data)[ptr_]; ptr_ += 20 * 20 * 20;
//
//    int min_sizes_16[2] = { 64, 128 };
//    std::vector<face_box> box_list_16(20 * 20 * 2);
//    const int faceCount16 = generate_anchors(16, min_sizes_16,
//        score_blob_16, bbox_blob_16, landmark_blob_16,
//        20, 20, 20, 20, 20, 20,
//        box_list_16.data(), target_size_, img_size_arr);
//
//    // -------- stride 8 (feature 10x10) --------
//    float* bbox_blob_8 = &((float*)output.data)[ptr_]; ptr_ += 8 * 10 * 10;
//    float* score_blob_8 = &((float*)output.data)[ptr_]; ptr_ += 4 * 10 * 10;
//    float* landmark_blob_8 = &((float*)output.data)[ptr_]; // last
//
//    int min_sizes_8[2] = { 256, 512 };
//    std::vector<face_box> box_list_8(10 * 10 * 2);
//    const int faceCount8 = generate_anchors(32, min_sizes_8,
//        score_blob_8, bbox_blob_8, landmark_blob_8,
//        10, 10, 10, 10, 10, 10,
//        box_list_8.data(), target_size_, img_size_arr);
//
//    // -------- �ϲ� + ���� + NMS --------
//    const int total_face_count = faceCount32 + faceCount16 + faceCount8;
//    if (total_face_count <= 0) return -3001;
//
//    std::vector<face_box> total(total_face_count);
//    // ֻ������Ч������faceCountXX������Ҫ�������� buffer
//    if (faceCount32 > 0) std::memcpy(total.data() + 0,
//        box_list_32.data(),
//        sizeof(face_box) * faceCount32);
//    if (faceCount16 > 0) std::memcpy(total.data() + faceCount32,
//        box_list_16.data(),
//        sizeof(face_box) * faceCount16);
//    if (faceCount8 > 0) std::memcpy(total.data() + faceCount32 + faceCount16,
//        box_list_8.data(),
//        sizeof(face_box) * faceCount8);
//
//    // ���Ŷ�����
//    if (!total.empty()) qsort_descent_inplace(total.data(), (int)total.size());
//
//    // NMS
//    std::vector<int> picked(total.size());
//    const int pickedCount = nms_sorted_bboxes(total.data(),
//        (int)total.size(),
//        picked.data(),
//        nms_threshold);
//
//    out_p_faces.clear();
//    out_p_faces.reserve(pickedCount);
//    for (int i = 0; i < pickedCount; ++i) {
//        face_box face = total[picked[i]];
//
//        // �������Ԥ�������320x320�ռ�ת����ԭʼͼ��ռ�
//        face.x0 = (face.x0 - left) / scale;
//        face.y0 = (face.y0 - top) / scale;
//        face.x1 = (face.x1 - left) / scale;
//        face.y1 = (face.y1 - top) / scale;
//
//        // �ؼ�������ת��
//        for (int j = 0; j < LANDMARK_NUMBER; j++) {
//            face.landmark.x[j] = (face.landmark.x[j] - left) / scale;
//            face.landmark.y[j] = (face.landmark.y[j] - top) / scale;
//        }
//
//        out_p_faces.push_back(face);
//    }
//    return 0;
//}

// ====================== ������� + 5����� ======================
typedef struct
{
	double a, b, c; // ��һ��Ԫ��
	double d, e, f; // �ڶ���Ԫ��
} AffineTransform;

typedef struct XtPointF {
	float x;  ///< ������
	float y;  ///< ������
} XtPointF;

typedef struct BBoxRect {
	float score;
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float area;
	int label;
} BBoxRect;

typedef struct {
	int landmarks_num;
	int* landmark_keys;
	XtPointF* landmarks2d;     ///< 2D�ؼ�������
	int bbox_2d[6];  // 2D������߽��˳��Ϊ[left, top, right, bottom, width, height]
	int w;
	int h;
	int c;
	unsigned char* image;
	unsigned char* extractImage;
	BBoxRect bbox;
} Xt3DFaceInfo;

double determinant(const AffineTransform* t)
{
	return t->a * t->e - t->b * t->d;
}

float bilinear_interpolation(float x, float y, float q11, float q12,
	float q21, float q22, float x1, float x2, float y1, float y2)
{
	float temp1 = (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21;
	float temp2 = (x2 - x) / (x2 - x1) * q12 + (x - x1) / (x2 - x1) * q22;
	return (y2 - y) / (y2 - y1) * temp1 + (y - y1) / (y2 - y1) * temp2;
}

// ������任���󣨴������ݴ���
int inverse_transform(const AffineTransform* src, AffineTransform* dst, double epsilon)
{
	const double det = determinant(src);
	if (fabs(det) < epsilon)
	{
		return 0; // ������
	}

	const double inv_det = 1.0 / det;

	// �������Բ��������
	dst->a = src->e * inv_det;
	dst->b = -src->b * inv_det;
	dst->d = -src->d * inv_det;
	dst->e = src->a * inv_det;

	// ����ƽ�Ʋ���������
	dst->c = (src->b * src->f - src->e * src->c) * inv_det;
	dst->f = (src->d * src->c - src->a * src->f) * inv_det;

	return 1;
}

// Ӧ����任����
XtPointF apply_inverse(const AffineTransform* t, const XtPointF* pt)
{
	XtPointF a;
	a.x = t->a * pt->x + t->b * pt->y + t->c;
	a.y = t->d * pt->x + t->e * pt->y + t->f;
	return a;
}


//void AffineTrans(const face_box& face_info, ncnn::Mat ori_img, ncnn::Mat& extractImage)
//{
//    //ncnn::Mat in = ncnn::Mat::from_pixels(face_info->image, ncnn::Mat::PIXEL_BGR, face_info->w, face_info->h);
//    int w = ori_img.w;
//    int h = ori_img.h;
//    ncnn::Mat in = ori_img;
//    
//    extractImage.create(96, 112, 3);
//    extractImage.fill(0);
//    XtPointF ptSrc[5];
//    for (int i = 0; i < 5; i++) {
//        ptSrc[i].x = face_info.landmark.x[i];
//        ptSrc[i].y = face_info.landmark.y[i];
//    }
//
//    float ptEyesMiddle0_x = (ptSrc[0].x + ptSrc[1].x) / 2;
//    float ptEyesMiddle0_y = (ptSrc[0].y + ptSrc[1].y) / 2;
//
//    float dyEyes0 = ptSrc[1].y - ptSrc[0].y;
//    float dxEyes0 = ptSrc[1].x - ptSrc[0].x;
//    float angle0 = atan2f(dyEyes0, dxEyes0) * 180.0 / CV_PI;
//    /// 2. scale
//    float ptMouthMiddle0_x = (ptSrc[3].x + ptSrc[4].x) / 2;
//    float ptMouthMiddle0_y = (ptSrc[3].y + ptSrc[4].y) / 2;
//    float dx = ptMouthMiddle0_x - ptEyesMiddle0_x;
//    float dy = ptMouthMiddle0_y - ptEyesMiddle0_y;
//
//    float dist = sqrtf(dx * dx + dy * dy);
//    // ec_mc_y
//    float scale = 41 / dist;
//    ncnn::Mat M(3, 2, 1);
//    ncnn::get_rotation_matrix(angle0, scale, ptEyesMiddle0_x, ptEyesMiddle0_y, M);
//    /// 3. translation
//    float ptEyesMiddle1_x = 96 / 2; // width
//    // ec_y
//    float ptEyesMiddle1_y = 51;
//
//    M.row(0)[2] += ptEyesMiddle1_x - ptEyesMiddle0_x;
//    M.row(1)[2] += ptEyesMiddle1_y - ptEyesMiddle0_y;
//
//    AffineTransform rotate;
//    rotate.a = M.row(0)[0];
//    rotate.b = M.row(0)[1];
//    rotate.c = M.row(0)[2];
//    rotate.d = M.row(1)[0];
//    rotate.e = M.row(1)[1];
//    rotate.f = M.row(1)[2];
//
//    AffineTransform inv_rotate;
//    if (inverse_transform(&rotate, &inv_rotate, 1e-8))
//    {
//        for (int x = 0; x < extractImage.w; x++)
//        {
//            for (int y = 0; y < extractImage.h; y++)
//            {
//                XtPointF rotated = { double(x), double(y) };
//                XtPointF original = apply_inverse(&inv_rotate, &rotated);
//
//                if (original.x >= w || original.y >= h || original.x <= 0 || original.y <= 0) {
//                    continue;
//                }
//                else {
//
//                    extractImage.channel(0).row(y)[x] = bilinear_interpolation(original.x, original.y, in.channel(0).row(floor(original.y))[int(floor(original.x))], in.channel(0).row(floor(original.y))[int(ceil(original.x))], in.channel(0).row(ceil(original.y))[int(floor(original.x))], in.channel(0).row(ceil(original.y))[int(ceil(original.x))], floor(original.x), ceil(original.y), floor(original.y), ceil(original.y));
//                    extractImage.channel(1).row(y)[x] = bilinear_interpolation(original.x, original.y, in.channel(1).row(floor(original.y))[int(floor(original.x))], in.channel(1).row(floor(original.y))[int(ceil(original.x))], in.channel(1).row(ceil(original.y))[int(floor(original.x))], in.channel(1).row(ceil(original.y))[int(ceil(original.x))], floor(original.x), ceil(original.y), floor(original.y), ceil(original.y));
//                    extractImage.channel(2).row(y)[x] = bilinear_interpolation(original.x, original.y, in.channel(2).row(floor(original.y))[int(floor(original.x))], in.channel(2).row(floor(original.y))[int(ceil(original.x))], in.channel(2).row(ceil(original.y))[int(floor(original.x))], in.channel(2).row(ceil(original.y))[int(ceil(original.x))], floor(original.x), ceil(original.y), floor(original.y), ceil(original.y));
//                }
//            }
//        }
//    }
//}


void AffineTrans(const face_box& face_info, ncnn::Mat ori_img, ncnn::Mat& extractImage)
{
	int w = ori_img.w;
	int h = ori_img.h;
	ncnn::Mat in = ori_img;

	extractImage.create(112, 112, 3);
	extractImage.fill(0);

	XtPointF ptSrc[5];
	for (int i = 0; i < 5; i++) {
		ptSrc[i].x = face_info.landmark.x[i];
		ptSrc[i].y = face_info.landmark.y[i];
	}

	float ptEyesMiddle0_x = (ptSrc[0].x + ptSrc[1].x) / 2;
	float ptEyesMiddle0_y = (ptSrc[0].y + ptSrc[1].y) / 2;

	float dyEyes0 = ptSrc[1].y - ptSrc[0].y;
	float dxEyes0 = ptSrc[1].x - ptSrc[0].x;
	float angle0 = atan2f(dyEyes0, dxEyes0) * 180.0 / CV_PI;
	/// 2. scale
	float ptMouthMiddle0_x = (ptSrc[3].x + ptSrc[4].x) / 2;
	float ptMouthMiddle0_y = (ptSrc[3].y + ptSrc[4].y) / 2;
	float dx = ptMouthMiddle0_x - ptEyesMiddle0_x;
	float dy = ptMouthMiddle0_y - ptEyesMiddle0_y;

	float dist = sqrtf(dx * dx + dy * dy);
	// ec_mc_y
	float scale = 41 / dist;
	ncnn::Mat M(3, 2, 1);
	ncnn::get_rotation_matrix(angle0, scale, ptEyesMiddle0_x, ptEyesMiddle0_y, M);
	/// 3. translation
	float ptEyesMiddle1_x = 112 / 2; // width
	// ec_y
	float ptEyesMiddle1_y = 45;

	M.row(0)[2] += ptEyesMiddle1_x - ptEyesMiddle0_x;
	M.row(1)[2] += ptEyesMiddle1_y - ptEyesMiddle0_y;

	AffineTransform rotate;
	rotate.a = M.row(0)[0];
	rotate.b = M.row(0)[1];
	rotate.c = M.row(0)[2];
	rotate.d = M.row(1)[0];
	rotate.e = M.row(1)[1];
	rotate.f = M.row(1)[2];

	AffineTransform inv_rotate;
	if (inverse_transform(&rotate, &inv_rotate, 1e-8))
	{
		// �߽�����˫���Բ�ֵ
		auto border_bilinear_interp = [&](float x, float y, int channel) -> float {
			// ʹ�ñ߽�����ֵ�������
			float clamped_x = std::max(0.0f, std::min(static_cast<float>(w - 1), x));
			float clamped_y = std::max(0.0f, std::min(static_cast<float>(h - 1), y));

			int x0 = static_cast<int>(clamped_x);
			int y0 = static_cast<int>(clamped_y);
			int x1 = std::min(w - 1, x0 + 1);
			int y1 = std::min(h - 1, y0 + 1);

			float dx = clamped_x - x0;
			float dy = clamped_y - y0;

			// �����߽��������x0==x1��y0==y1ʱ
			if (x0 == x1) dx = 0.0f;
			if (y0 == y1) dy = 0.0f;

			float q11 = in.channel(channel).row(y0)[x0];
			float q21 = in.channel(channel).row(y0)[x1];
			float q12 = in.channel(channel).row(y1)[x0];
			float q22 = in.channel(channel).row(y1)[x1];

			float value =
				q11 * (1 - dx) * (1 - dy) +
				q21 * dx * (1 - dy) +
				q12 * (1 - dx) * dy +
				q22 * dx * dy;

			return value;
		};

		for (int x = 0; x < extractImage.w; x++)
		{
			for (int y = 0; y < extractImage.h; y++)
			{
				XtPointF rotated = { static_cast<double>(x), static_cast<double>(y) };
				XtPointF original = apply_inverse(&inv_rotate, &rotated);

				extractImage.channel(0).row(y)[x] = border_bilinear_interp(original.x, original.y, 0);
				extractImage.channel(1).row(y)[x] = border_bilinear_interp(original.x, original.y, 1);
				extractImage.channel(2).row(y)[x] = border_bilinear_interp(original.x, original.y, 2);
			}
		}
	}
}






static ncnn::Mat resize_with_padding(const ncnn::Mat& image, int target_width, int target_height, const ncnn::Mat& padding_color)
{
	// ȷ��������Ч
	if (image.empty() || target_width <= 0 || target_height <= 0) {
		return ncnn::Mat();
	}

	int w = image.w;
	int h = image.h;
	int c = image.c;

	// �������ű���
	// �������ű���
	float scale = std::min(target_width / (float)w, target_height / (float)h);
	int new_w = std::max(1, (int)std::round(w * scale));
	int new_h = std::max(1, (int)std::round(h * scale));

	// ʹ��˫���Բ�ֵ����ͼ�񣨱���ԭʼͨ��˳��
	ncnn::Mat resized;
	ncnn::resize_bilinear(image, resized, new_w, new_h);

	// ����Ŀ�����
	ncnn::Mat padded(target_width, target_height, c);

	// ʹ��padding_color��䣨����padding_color��BGR˳��
	for (int q = 0; q < c; q++) {
		float fill_value = padding_color[q]; // ��ͨ����ȡ���ֵ
		float* ptr = padded.channel(q);
		for (int i = 0; i < target_width * target_height; i++) {
			ptr[i] = fill_value;
		}
	}

	// �������λ��
	int top = (target_height - new_h) / 2;
	int left = (target_width - new_w) / 2;

	// �������ź��ͼ�����루����BGR˳��
	for (int q = 0; q < c; q++) {
		const float* rptr = resized.channel(q);
		float* pptr = padded.channel(q);

		// ����ȫ��ָ����㷽��
		for (int y = 0; y < new_h; y++) {
			const float* inptr = rptr + y * new_w;
			float* outptr = pptr + ((y + top) * target_width) + left;
			for (int x = 0; x < new_w; x++) {
				outptr[x] = inptr[x];
			}
		}
	}

	return padded;
}


class faceDetectModel {
public:
	faceDetectModel(const std::string& param_path, const std::string& bin_path) {
		net.opt.use_int8_inference = true;
		net.load_param(param_path.c_str());
		net.load_model(bin_path.c_str());
	}
	// �� face_detect.cpp һ�£������� "input"������� "output"
	//ncnn::Mat predict(const cv::Mat& img_bgr_320x320_padded) {
	ncnn::Mat predict(const ncnn::Mat& img_bgr_320x320_padded) {
		ncnn::Mat in = img_bgr_320x320_padded;
		const float mean_vals[3] = { 104.f, 117.f, 123.f };
		const float norm_vals[3] = { 1.f, 1.f, 1.f };
		in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = net.create_extractor();
		ex.input("input", in);
		ncnn::Mat out;
		ex.extract("output", out);
		return out;
	}
private:
	ncnn::Net net;
};


class faceRecModel {
public:
	faceRecModel(const std::string& param_path, const std::string& bin_path) {
		net.opt = ncnn::Option();
		net.opt.use_int8_inference = true;
		net.load_param(param_path.c_str());
		net.load_model(bin_path.c_str());
	}

	//ncnn::Mat predict(const cv::Mat& img) 
	ncnn::Mat predict(const ncnn::Mat& img)
	{
		ncnn::Mat in = img;

		// ��һ��
		const float mean_vals[3] = { 127.5, 127.5, 127.5 };
		const float norm_vals[3] = { 0.007843f, 0.007843f, 0.007843f };
		in.substract_mean_normalize(mean_vals, norm_vals);

		// ǰ�����
		ncnn::Extractor ex = net.create_extractor();
		ex.input("input", in);  // ע����ONNX����ʱ��������һ��

		ncnn::Mat out;
		ex.extract("output", out);    // ��ONNX����ʱ�������һ��

		return out;
	}

private:
	ncnn::Net net;
};

std::vector<float> ncnnMatToVector(const ncnn::Mat& mat) {
	std::vector<float> vec(mat.w);
	for (int i = 0; i < mat.w; i++) {
		vec[i] = mat[i];
	}
	return vec;
}

static float CompareFeatures(const std::vector<float>& feature1,
	const std::vector<float>& feature2)
{
	if (feature1.size() != feature2.size())
	{
		return 0.0f;
	}

	float dot = 0.0f;
	float norm1 = 0.0f;
	float norm2 = 0.0f;

	for (unsigned int i = 0; i < feature1.size(); i++)
	{
		dot += feature1[i] * feature2[i];
		norm1 += feature1[i] * feature1[i];
		norm2 += feature2[i] * feature2[i];
	}

	if (norm1 == 0.0f || norm2 == 0.0f)
	{
		return 0.0f; // ���������
	}

	return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}


void ncnn_to_cv_save(ncnn::Mat& ncnn_mat, const std::string& save_path) {
	// ��ͼʹ��
	ncnn::Mat uint8_roi;
	uint8_roi.create(ncnn_mat.w, ncnn_mat.h, ncnn_mat.c);
	// �ֶ�ת�� float �� uint8
	for (int c = 0; c < ncnn_mat.c; c++) {
		const float* src = ncnn_mat.channel(c);
		unsigned char* dst = uint8_roi.channel(c);
		const int size = ncnn_mat.w * ncnn_mat.h;
		for (int i = 0; i < size; i++) {
			float val = src[i];
			val = std::min(std::max(val, 0.0f), 255.0f); // ȷ���� [0,255] ��Χ��
			dst[i] = static_cast<unsigned char>(val + 0.5f); // ��������
		}
	}

	// ��ƽ���ʽת��Ϊ������ʽ
	cv::Mat cv_roi(uint8_roi.h, uint8_roi.w, CV_8UC3);
	std::vector<cv::Mat> channels;
	for (int i = 0; i < uint8_roi.c; i++) {
		channels.push_back(cv::Mat(uint8_roi.h, uint8_roi.w, CV_8UC1, uint8_roi.channel(i)));
	}
	cv::merge(channels, cv_roi);
	uint8_roi.release();
	cv::imwrite(save_path, cv_roi);
}


void normalize_l2(std::vector<float>& feature)
{
	float norm = 0.0f;

	// ���� L2 ����
	for (float val : feature)
	{
		norm += val * val;
	}

	norm = std::sqrt(norm);

	// ��ֹ������
	if (norm > 1e-8f)
	{
		for (float& val : feature)
		{
			val /= norm;
		}
	}
}


float bilinear_interpolate_yuv(const unsigned char* src_plane, int src_w, int src_h, float x, float y) {
	if (x < 0 || x >= src_w - 1 || y < 0 || y >= src_h - 1) {
		return 0.0f; // Return a default value for out-of-bounds access
	}

	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	float fx = x - x1;
	float fy = y - y1;

	float val11 = static_cast<float>(src_plane[y1 * src_w + x1]);
	float val12 = static_cast<float>(src_plane[y1 * src_w + x2]);
	float val21 = static_cast<float>(src_plane[y2 * src_w + x1]);
	float val22 = static_cast<float>(src_plane[y2 * src_w + x2]);

	float interpolated_val = (1.0f - fx) * (1.0f - fy) * val11 +
		fx * (1.0f - fy) * val12 +
		(1.0f - fx) * fy * val21 +
		fx * fy * val22;

	return interpolated_val;
}

//unsigned char* get_new_box_our(int src_w, int src_h, float x, float y, float box_w, float box_h, float scale, const unsigned char* ori) {
//
//    // ���� scale����ֹĿ��򳬹�ͼ��ߴ�
//    float max_scale_w = static_cast<float>(src_w) / box_w;
//    float max_scale_h = static_cast<float>(src_h) / box_h;
//    scale = std::min({ scale, max_scale_w, max_scale_h });
//
//    // ���ź����
//    float target_len_float = std::max(box_w, box_h) * scale;
//
//    // ���ĵ�
//    float center_x = x + box_w / 2.0f;
//    float center_y = y + box_h / 2.0f;
//
//    // �����¿�����ϽǺ����½�
//    float left_top_x = center_x - target_len_float / 2.0f;
//    float left_top_y = center_y - target_len_float / 2.0f;
//    float right_bottom_x = center_x + target_len_float / 2.0f;
//    float right_bottom_y = center_y + target_len_float / 2.0f;
//
//    // ������ֹԽ��
//    if (left_top_x < 0) {
//        right_bottom_x -= left_top_x;
//        left_top_x = 0;
//    }
//    if (left_top_y < 0) {
//        right_bottom_y -= left_top_y;
//        left_top_y = 0;
//    }
//    if (right_bottom_x > src_w - 1) {
//        left_top_x -= (right_bottom_x - (src_w - 1));
//        right_bottom_x = static_cast<float>(src_w - 1);
//    }
//    if (right_bottom_y > src_h - 1) {
//        left_top_y -= (right_bottom_y - (src_h - 1));
//        right_bottom_y = static_cast<float>(src_h - 1);
//    }
//
//    // ���յ������߽��
//    int final_x = static_cast<int>(std::round(left_top_x));
//    int final_y = static_cast<int>(std::round(left_top_y));
//    int final_w = static_cast<int>(std::round(right_bottom_x - left_top_x));
//    int final_h = static_cast<int>(std::round(right_bottom_y - left_top_y));
//
//    // ȷ���ü���Խ��
//    final_x = std::max(0, final_x);
//    final_y = std::max(0, final_y);
//    final_w = std::min(src_w - final_x, final_w);
//    final_h = std::min(src_h - final_y, final_h);
//
//    if (final_w <= 0 || final_h <= 0) {
//        return nullptr;
//    }
//
//    // ��ȡԭʼ YUV ƽ������ָ��
//    int src_y_size = src_w * src_h;
//    const unsigned char* src_y_plane = ori;
//    const unsigned char* src_u_plane = ori + src_y_size;
//    const unsigned char* src_v_plane = ori + src_y_size + (src_w / 2) * (src_h / 2);
//
//    // Ŀ��ͼ��ߴ�
//    const int new_w = 80;
//    const int new_h = 80;
//    const int new_y_size = new_w * new_h;
//    const int new_uv_w = new_w / 2;
//    const int new_uv_h = new_h / 2;
//    const int new_uv_size = new_uv_w * new_uv_h;
//    const int new_total_size = new_y_size + new_uv_size * 2;
//
//    // �������ڴ�
//    unsigned char* new_image_data = new unsigned char[new_total_size];
//    unsigned char* new_y_plane = new_image_data;
//    unsigned char* new_u_plane = new_image_data + new_y_size;
//    unsigned char* new_v_plane = new_image_data + new_y_size + new_uv_size;
//
//    // 1. Resize Y plane
//    float scale_x_y = static_cast<float>(final_w) / new_w;
//    float scale_y_y = static_cast<float>(final_h) / new_h;
//
//    for (int y_dest = 0; y_dest < new_h; ++y_dest) {
//        for (int x_dest = 0; x_dest < new_w; ++x_dest) {
//            float x_src = x_dest * scale_x_y + final_x;
//            float y_src = y_dest * scale_y_y + final_y;
//            new_y_plane[y_dest * new_w + x_dest] = static_cast<unsigned char>(bilinear_interpolate_yuv(src_y_plane, src_w, src_h, x_src, y_src));
//        }
//    }
//
//    // 2. Resize U and V planes
//    float scale_x_uv = static_cast<float>(final_w / 2.0f) / new_uv_w;
//    float scale_y_uv = static_cast<float>(final_h / 2.0f) / new_uv_h;
//    int src_uv_w = src_w / 2;
//    int src_uv_h = src_h / 2;
//    int final_uv_x = final_x / 2;
//    int final_uv_y = final_y / 2;
//
//    for (int y_dest = 0; y_dest < new_uv_h; ++y_dest) {
//        for (int x_dest = 0; x_dest < new_uv_w; ++x_dest) {
//            float x_src = x_dest * scale_x_uv + final_uv_x;
//            float y_src = y_dest * scale_y_uv + final_uv_y;
//
//            new_u_plane[y_dest * new_uv_w + x_dest] = static_cast<unsigned char>(bilinear_interpolate_yuv(src_u_plane, src_uv_w, src_uv_h, x_src, y_src));
//            new_v_plane[y_dest * new_uv_w + x_dest] = static_cast<unsigned char>(bilinear_interpolate_yuv(src_v_plane, src_uv_w, src_uv_h, x_src, y_src));
//        }
//    }
//
//    return new_image_data;
//}


// 0202�汾�������βü���
cv::Mat get_new_box_our(int src_w, int src_h, float x, float y, float box_w, float box_h, float scale, cv::Mat ori) {

	// ���� scale����ֹĿ��򳬹�ͼ��ߴ�
	float max_scale_w = (src_w - 1.0f) / box_w;
	float max_scale_h = (src_h - 1.0f) / box_h;
	scale = std::min({ scale, max_scale_w, max_scale_h });

	// �Գ���Ϊ��׼����Ŀ�������α߳�
	int target_len = static_cast<int>(std::max(box_w, box_h) * scale);

	// ���ĵ�
	float center_x = x + box_w / 2.0f;
	float center_y = y + box_h / 2.0f;

	// �����ʼ�����α߽磨�����ĵ�Ϊ���ģ�
	int left_top_x = static_cast<int>(center_x - target_len / 2.0f);
	int left_top_y = static_cast<int>(center_y - target_len / 2.0f);
	int right_bottom_x = left_top_x + target_len;
	int right_bottom_y = left_top_y + target_len;

	// �����߽磬ȷ����������ͼ��Χ��
	// ��������γ����߽磬�����λ��ʹ���ڱ߽���
	if (left_top_x < 0) {
		left_top_x = 0;
		right_bottom_x = std::min(src_w, target_len);
	}
	if (left_top_y < 0) {
		left_top_y = 0;
		right_bottom_y = std::min(src_h, target_len);
	}
	if (right_bottom_x > src_w) {
		right_bottom_x = src_w;
		left_top_x = std::max(0, src_w - target_len);
	}
	if (right_bottom_y > src_h) {
		right_bottom_y = src_h;
		left_top_y = std::max(0, src_h - target_len);
	}

	// ����ȷ���������Σ�ȡ��С��һ����Ϊ�߳���
	int final_len = std::min(right_bottom_x - left_top_x, right_bottom_y - left_top_y);

	// ���µ���������λ�ã�ʹ���ڱ߽���
	// ���ȱ������ĵ�λ�ã�������������
	int final_x = left_top_x;
	int final_y = left_top_y;

	// ����Ƿ�Խ�粢����
	if (final_x + final_len > src_w) {
		final_x = src_w - final_len;
	}
	if (final_y + final_len > src_h) {
		final_y = src_h - final_len;
	}

	// ȷ���Ǹ�
	final_x = std::max(0, final_x);
	final_y = std::max(0, final_y);

	// ���ȷ�������δ�С������ͼ��߽�
	final_len = std::min(final_len, src_w - final_x);
	final_len = std::min(final_len, src_h - final_y);

	// �ü� ROI ������
	cv::Rect roi(final_x, final_y, final_len, final_len);
	return ori(roi).clone();
}




// ====================== Image file collection (OpenCV glob, no <filesystem>) ======================
// Supported image extensions (lowercase)
static const std::vector<std::string> kImageExtensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"
};

static inline std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static inline bool has_image_extension(const std::string& path) {
    // Find last dot after last slash/backslash
    const size_t slash_pos = path.find_last_of("/\\");
    const size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos) return false;
    if (slash_pos != std::string::npos && dot_pos < slash_pos) return false;

    std::string ext = to_lower_copy(path.substr(dot_pos));
    for (const auto& e : kImageExtensions) {
        if (ext == e) return true;
    }
    return false;
}

// Recursively collect all files under base_dir and filter by extension.
// Notes:
// - cv::glob is available in OpenCV 3/4.
// - Use pattern base_dir + "/*" with recursive=true to walk subfolders.
static std::vector<std::string> get_all_images_cvglob(const std::string& base_dir) {
    std::vector<cv::String> all;
    std::string pattern = base_dir;
    if (!pattern.empty()) {
        char last = pattern.back();
        if (last == '/' || last == '\\') {
            pattern += "*";
        } else {
            pattern += "/*";
        }
    }

    // If base_dir is empty, this will return empty results.
    cv::glob(pattern, all, /*recursive=*/true);

    std::vector<std::string> images;
    images.reserve(all.size());
    for (const auto& p : all) {
        std::string s = p;
        if (has_image_extension(s)) {
            images.push_back(s);
        }
    }
    return images;
}


int main() {

//	faceDetectModel detector("../model/face_detector_320x320_20260304_t2.sim.param", "../model/old/face_detector_320x320_20260304_t2.sim.bin");
	faceDetectModel detector("../model/old/flaten_test_20250801.sim.param", "../model/flaten_test_20250801.sim.bin");
	//faceRecModel feature_extractor("2dface_112x112_1120_opset11_rgb.param", "2dface_112x112_1120_opset11_rgb.bin");

	std::string base_dir = "../test";

	std::vector<std::string> img_path = get_all_images_cvglob(base_dir);

	std::ofstream file("./dec_SN+test.txt");

	if (file.is_open()) {



		for (int i = 0; i < img_path.size(); i++) {

			// ��ȡͼƬ
			cv::Mat img1 = cv::imread(img_path[i]);


			if (img1.empty()) {
				std::cerr << "[ERROR] Failed to read image: " << img_path[i] << std::endl;
				return -1;
			}


			ncnn::Mat img11 = ncnn::Mat::from_pixels(img1.data, ncnn::Mat::PIXEL_BGR, img1.cols, img1.rows);





			// Ԥ������ 320x320���ȱ� + padding����ֵ���� ncnn ������
			//cv::Mat img_pad1 = resize_with_padding(img1, 320, 320, cv::Scalar(104, 117, 123));
			//cv::Mat img_pad2 = resize_with_padding(img2, 320, 320, cv::Scalar(104, 117, 123));

			ncnn::Mat my_color(3, 1, 1);
			my_color[0] = 104.f;   // B
			my_color[1] = 117.f;   // G
			my_color[2] = 123.f; // R

			ncnn::Mat img_pad1 = resize_with_padding(img11, 320, 320, my_color);

			// �������
			ncnn::Mat det1 = detector.predict(img_pad1);

			std::vector<face_box> faceboxs1;
			detect_retinaface_forward(img11, det1, faceboxs1);
			std::cout << "���Ƽ�⵽��" << faceboxs1.size() << "��������" << std::endl;


			if (faceboxs1.empty()) {
				// δ��⵽������д��0,0,0,0����
				std::cout << img_path[i] << "," << "x0:0" << "y0:0" << "x1:0" << "y1:0" << " (δ��⵽����)" << std::endl;
				file << img_path[i] << ",0,0,0,0" << std::endl;
			}
			else {




				//������� Facebox�������Ϣ
				// (x0,y0)��������������㣻 (x1,y1)���������������
				std::cout << img1.cols << std::endl;
				for (int j = 0; j < faceboxs1.size(); j++) {
					cv::rectangle(img1, cv::Rect(cv::Point(faceboxs1[j].x0, faceboxs1[j].y0), cv::Size(faceboxs1[j].x1 - faceboxs1[j].x0, faceboxs1[j].y1 - faceboxs1[j].y0)), cv::Scalar(255, 0, 0), 0);

					int x00 = std::round(faceboxs1[j].x0);
					int y00 = std::round(faceboxs1[j].y0);
					int x11 = std::round(faceboxs1[j].x1);
					int y11 = std::round(faceboxs1[j].y1);

					std::cout << img_path[i] << "," << "x0:" << x00 << "y0:" << y00 << "x1:" << x11 << "y1:" << y11 << std::endl;

					file << img_path[i] << "," << x00 << "," << y00 << "," << x11 << "," << y11 << std::endl;


				}
				std::string suffix = "test" + std::to_string(i) + ".jpg";

				// cv::imshow("test1", img1);
				// cv::Mat img_copy = img1.clone();
				// cv::waitKey(0);
				// cv::imwrite("D:\\models\\ArcSoft_ArcFace_Java_Windows_x64_V3.0\\test\\" + suffix, img_copy);
				cv::rectangle(img1, cv::Rect(cv::Point(faceboxs1[0].x0, faceboxs1[0].y0), cv::Size(faceboxs1[0].x1 - faceboxs1[0].x0, faceboxs1[0].y1 - faceboxs1[0].y0)), cv::Scalar(255, 0, 0), 0);
				cv::imwrite("test2.jpg", img1);

			}





			//std::vector<std::vector<float>> features1;
			//std::vector<std::vector<float>> features2;


			//for (const auto& det : faceboxs1) {
			//    //cv::Mat dst1;
			//    //XtPointF lmk10 = { det.landmark.x[0], det.landmark.y[0] };
			//    //XtPointF lmk10 = { det.landmark.x[0], det.landmark.y[0] };
			//    //XtPointF lmk11 = { det.landmark.x[1], det.landmark.y[1] };
			//    //XtPointF lmk12 = { det.landmark.x[2], det.landmark.y[2] };
			//    //XtPointF lmk13 = { det.landmark.x[3], det.landmark.y[3] };
			//    //XtPointF lmk14 = { det.landmark.x[4], det.landmark.y[4] };
			//    //FaceAlign(dst1, img11, lmk10, lmk11, lmk12, lmk13, lmk14, 96, 112, 41, 51);
			//    //cv::resize(dst1, dst1, cv::Size(112, 112));

			//    ncnn::Mat dst1;
			//    AffineTrans(det, img11, dst1);

			//    //ncnn_to_cv_save(dst1, "dst1.jpg");
			//   // ncnn::Mat dst11;
			//   // ncnn::resize_bilinear(ccdst1, dst11, 112, 112);

			//    unsigned char resiz1[112 * 112 * 3];
			//    dst1.to_pixels(resiz1, ncnn::Mat::PIXEL_RGB);
			//    ncnn::Mat dst11 = ncnn::Mat::from_pixels_resize(resiz1, ncnn::Mat::PIXEL_BGR, dst1.w, dst1.h, 112, 112);
			//    ncnn::Mat face_feature1 = feature_extractor.predict(dst11);

			//    features1.push_back(ncnnMatToVector(face_feature1));
			//    //break;
			//}


			//for (const auto& det : faceboxs2) {
			//    //cv::Mat dst2;
			//    //XtPointF lmk20 = { det.landmark.x[0], det.landmark.y[0] };
			//    //XtPointF lmk21 = { det.landmark.x[1], det.landmark.y[1] };
			//    //XtPointF lmk22 = { det.landmark.x[2], det.landmark.y[2] };
			//    //XtPointF lmk23 = { det.landmark.x[3], det.landmark.y[3] };
			//    //XtPointF lmk24 = { det.landmark.x[4], det.landmark.y[4] };
			//    //FaceAlign(dst2, img22, lmk20, lmk21, lmk22, lmk23, lmk24, 96, 112, 41, 51);
			//    //cv::resize(dst2, dst2, cv::Size(112, 112));

			//    ncnn::Mat dst2;
			//    AffineTrans(det, img22, dst2);

			//    //ncnn_to_cv_save(dst2, "dst2.jpg");
			//    //ncnn::Mat dst22;
			//    //ncnn::resize_bilinear(ccdst2, dst22, 112, 112);

			//    unsigned char resiz2[112 * 112 * 3];
			//    dst2.to_pixels(resiz2, ncnn::Mat::PIXEL_RGB);
			//    ncnn::Mat dst22 = ncnn::Mat::from_pixels_resize(resiz2, ncnn::Mat::PIXEL_BGR, dst2.w, dst2.h, 112, 112);
			//    ncnn::Mat face_feature2 = feature_extractor.predict(dst22);

			//    features2.push_back(ncnnMatToVector(face_feature2));
			//    //break;
			//}

			//float score = -2;
			//if ((features1.size() != 0) && (features2.size() != 0)) {
			//    score = CompareFeatures(features1[0], features2[0]);
			//    //std::vector<float> fea1 = features1[0];
			//    //std::vector<float> fea2 = features2[0];
			//    //normalize_l2(fea1);
			//    //normalize_l2(fea2);
			//    //score = CompareFeatures(fea1, fea2);
			//}
		}
		std::cout << "face_box��ȡ����" << endl;
	}
	return 0;
}