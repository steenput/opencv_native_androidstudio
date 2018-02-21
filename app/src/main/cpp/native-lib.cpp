#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

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

}
