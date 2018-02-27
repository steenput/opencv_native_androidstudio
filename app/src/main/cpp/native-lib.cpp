#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

typedef cv::Matx<double, 3, 3> Mat33d;

extern "C"
{
// On doit nommer cette fonction selon le package, avec "J" à "Java", suivi du nom de la fonction
void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                           jlong matAddrGray,
                                                                           jint nbrElem) {
    Mat &mGr = *(Mat *) matAddrGray;
    for (int k = 0; k < nbrElem; k++) {
        int i = rand() % mGr.cols;
        int j = rand() % mGr.rows;
        mGr.at<uchar>(j, i) = 255;
    }
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_binary(JNIEnv *env, jobject instance,
                                                                           jlong matAddrGray) {
    Mat &mGr = *(Mat *) matAddrGray;
    for (int rows = 0; rows < mGr.rows; rows++) {
        for (int cols = 0; cols < mGr.cols; cols++) {
            if (mGr.at<uchar>(rows, cols) > 127) {
                mGr.at<uchar>(rows, cols) = 255;
            }
            else {
                mGr.at<uchar>(rows, cols) = 0;
            }
        }
    }
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_reduce(JNIEnv *env, jobject instance,
                                                                             jlong matAddrGray, jint n) {
    Mat &mGr = *(Mat *) matAddrGray;

    // accept only char type matrices
    CV_Assert(mGr.depth() == CV_8U);

    const int channels = mGr.channels();
    switch (channels) {
        case 1: {
            MatIterator_<uchar> it, end;
            for (it = mGr.begin<uchar>(), end = mGr.end<uchar>(); it != end; ++it)
                *it = (uchar) ((*it / n) * n + n / 2);
            break;
        }
        case 4: {
            MatIterator_<Vec3b> it, end;
            for (it = mGr.begin<Vec3b>(), end = mGr.end<Vec3b>(); it != end; ++it) {
                (*it)[0] = (uchar) (((*it)[0] / n) * n + n / 2);
                (*it)[1] = (uchar) (((*it)[1] / n) * n + n / 2);
                (*it)[2] = (uchar) (((*it)[2] / n) * n + n / 2);
                (*it)[3] = (uchar) (((*it)[3] / n) * n + n / 2);
            }
        }
    }
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_accentuation(JNIEnv *env, jobject instance,
                                                                             jlong matAddrGray, jlong matAddrAccent, jint radius) {
    Mat &matDst = *(Mat *) matAddrAccent;
    Mat &matSrc = *(Mat *) matAddrGray;

    int channels = matSrc.channels();
    int nRows = matSrc.rows;
    int nCols = matSrc.cols * channels;

    uchar *pSrc, *pSrcPrec, *pSrcNext, *pDst;
    for(int i = 1; i < nRows - 1; ++i) {
        pSrcPrec = matSrc.ptr<uchar>(i - 1);
        pSrc = matSrc.ptr<uchar>(i);
        pSrcNext = matSrc.ptr<uchar>(i + 1);
        pDst = matDst.ptr<uchar>(i);
        for (int j = 1; j < nCols - 1; ++j) {
            pDst[j] = saturate_cast<uchar>(pSrc[j] * radius - pSrc[j - 1] - pSrc[j + 1] - pSrcPrec[j] - pSrcNext[j]);
        }
    }
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_accentuation2(JNIEnv *env, jobject instance,
                                                                                    jlong src,
                                                                                    jlong dst, jint radius) {
    Mat &matDst = *(Mat *) dst;
    Mat &matSrc = *(Mat *) src;

    CV_Assert(matSrc.depth() == CV_8U);
    Mat33d kern(0, -1, 0,
                -1, radius, -1,
                0, -1, 0);

    filter2D(matSrc, matDst, matSrc.depth(), kern);
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_blur(JNIEnv *env, jobject instance,
                                                                           jlong src, jlong dst,
                                                                           jdouble level) {
    Mat &matDst = *(Mat *) dst;
    Mat &matSrc = *(Mat *) src;

    CV_Assert(matSrc.depth() == CV_8U);
    Mat33d kern(0.0, level, 0.0,
                level, level, level,
                0.0, level, 0.0);

    filter2D(matSrc, matDst, matSrc.depth(), kern);
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_blur2(JNIEnv *env, jobject instance,
                                                                           jlong src, jlong dst,
                                                                           jint level) {
    Mat &matDst = *(Mat *) dst;
    Mat &matSrc = *(Mat *) src;
    medianBlur(matSrc, matDst, level);
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_laplacian(JNIEnv *env, jobject instance,
                                                                            jlong src, jlong dst) {
    Mat &matDst = *(Mat *) dst;
    Mat &matSrc = *(Mat *) src;
    Laplacian(matSrc, matDst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(matDst, matDst);
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_threshold(JNIEnv *env, jobject instance,
                                                                                jlong src, jlong dst, jint thresholdValue) {
    Mat &matDst = *(Mat *) dst;
    Mat &matSrc = *(Mat *) src;
    CV_Assert(matSrc.depth() == CV_8U);
    threshold(matSrc, matDst, thresholdValue, 255, 0);
}

void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_inversion(JNIEnv *env, jobject instance,
                                                                                jlong matAddr,
                                                                                jint thresholdVal) {
    Mat &mGr = *(Mat *) matAddr;
    MatIterator_<uchar> it, end;
    for( it = mGr.begin<uchar>(), end = mGr.end<uchar>(); it != end; ++it) {
        if (*it < thresholdVal) *it = 255;
        else *it = 0;
    }
}

}
