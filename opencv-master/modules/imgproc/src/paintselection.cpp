//
//  paintselection.cpp
//  
//
//  Created by vedanshu kumar on 07/03/19.
//

/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                        Intel License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000, Intel Corporation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of Intel Corporation may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include "gcgraph.hpp"
#include <limits>
#include <time.h>
#include <mutex>
#include <thread>

#include <stdlib.h>
using namespace cv;

/*
 This is implementation of image segmentation algorithm GrabCut described in
 "GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
 Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
 */
class GMM
{
public:
    int componentsCount;
    
    GMM( Mat& _model, int cCount);
    ~GMM();
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;
    
    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();
    
private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;
    
    double*** inverseCovs;
    double* covDeterms;
    
    double** sums;
    double*** prods;
    int* sampleCounts;
    int totalSampleCount;
};

GMM::GMM( Mat& _model, int cCount = 5)
{
    componentsCount = cCount;
    inverseCovs = new double**[cCount];
    covDeterms = new double[cCount];
    sums = new double*[cCount];
    prods = new double**[cCount];
    sampleCounts = new int[cCount];
    
    for(size_t i=0; i < cCount; ++i){
        inverseCovs[i] = new double*[3];
        sums[i] = new double[3];
        prods[i] = new double*[3];
        for(size_t j=0; j < 3; ++j){
            inverseCovs[i][j] = new double[3];
            prods[i][j] = new double[3];
        }
    }
    
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );
    
    model = _model;
    
    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;
    
    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
            calcInverseCovAndDeterm( ci );
}

GMM::~GMM(){
    for(size_t i=0; i < componentsCount; ++i){
        for(size_t j=0; j < 3; ++j){
            delete [] inverseCovs[i][j];
            delete [] prods[i][j];
        }
        delete [] inverseCovs[i];
        delete [] sums[i];
        delete [] prods[i];
    }
    delete [] inverseCovs;
    delete [] covDeterms;
    delete [] sums;
    delete [] prods;
    delete [] sampleCounts;
}

double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
        + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
        + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;
    
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
    
}

void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;
            
            double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;
            
            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];
            
            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }
            
            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm =
        covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
        
        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

/*
 Calculate beta - parameter of GrabCut algorithm.
 beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
 */
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    
    return beta;
}

/*
 Calculate weights of noterminal vertices of graph.
 beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
 Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                         "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
 Initialize mask using rectangular.
 */
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );
    
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);
    
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
 Initialize GMM background and foreground models using kmeans algorithm.
 */
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
    
    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, bgdGMM.componentsCount, bgdLabels,
           TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, fgdGMM.componentsCount, fgdLabels,
           TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    
    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();
    
    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

/*
 Assign GMMs components for each pixel.
 */
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
            bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
 Learn GMMs parameters.
 */
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < bgdGMM.componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    for( int ci = 0; ci < fgdGMM.componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ){}
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
 multithread stuff
 */


#define r_split 8

// regions in image
#define r_count r_split*r_split

std::mutex m;

// shared index for task queue
int current_region = 0;

static void worker(GCGraph<double> * graph, double * result, int f)
{
    int region;
    
    for (;;)
    {
        std::unique_lock<std::mutex> lk(m);
        region = current_region++;
        lk.unlock();
        if (region >= r_count)
            break;
        result[region] = graph->maxFlow(region, f);
    }
}



/*
 Construct partially reduced GCGraph.
 Pixels marked as BG or FG are merged with terminal nodes. The Mat pxl2Vtx
 records the index of vertex for each pixel.
 To enable parallel computation of max Flow, the image is partitionned into
 regions, and each vertex is indexed by the corresponding region number.
 */
static void constructGCGraph_slim( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                                  const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                                  GCGraph<double>& graph, Mat& pxl2Vtx)
{
    int vtxCount = img.cols*img.rows,
    edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    
    // region numbering
#define TRANS 200
    //int h_size = (img.cols+TRANS) / r_split + 1, v_size = (img.rows + TRANS)/ r_split + 1;
    int h_size = img.cols / r_split + 1, v_size = img.rows  / r_split + 1;
    
    int r_split2 = r_split - 1;
    int h_size2 = img.cols / r_split2 + 1, v_size2 = img.rows / r_split2 + 1;
    
    std::vector<std::vector<int>> r_index(r_split, std::vector<int>(r_split));
    std::vector<std::vector<int>> r_index2(r_split, std::vector<int>(r_split));
    
    //quadtree(QT_HEIGHT, r_index);
    for (int i = 0; i < r_split; i++)
        for (int j = 0; j < r_split; j++)
            r_index[i][j] = i*r_split + j;
    
    for (int i = 0; i < r_split2; i++)
        for (int j = 0; j < r_split2; j++)
            r_index2[i][j] = i*r_split2 + j;
    graph.create(vtxCount, edgeCount);
    Point p;
    //int vtxIdx;
    
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            Vec3b color = img.at<Vec3b>(p);
            
            // add node and set its t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                int r = r_index[p.y / v_size][p.x / h_size];
                int r1 = r_index[p.y / v_size][(p.x + TRANS) / h_size];
                int r2 = r_index[(p.y + TRANS) / v_size][p.x / h_size];
                int r3 = r_index[(p.y + TRANS) / v_size][(p.x + TRANS) / h_size];
                int alt_r=r;
                if (r != r1)
                    if (r == r2)
                        alt_r = r1;
                    else
                        alt_r = r3;
                    else
                        if (r != r2)
                            alt_r = r2;
                alt_r = r_index[p.y / v_size2][p.x / h_size2];
                int vtxIdx = graph.addVtx(r, alt_r);
                pxl2Vtx.at<int>(p) = vtxIdx;
                fromSource = -log(bgdGMM(color));
                toSink = -log(fgdGMM(color));
                graph.addTermWeights(vtxIdx, fromSource, toSink);
            }
            else if( mask.at<uchar>(p) == GC_BGD )
                // join to sink
            {
                pxl2Vtx.at<int>(p) = GC_JNT_BGD; // join node to sink (BG) -1 = GC_JNT_BGD
                //fromSource = 0;  // as fromSource=0, weight of edge(source, sink) is unmodified
                //toSink = lambda; // edge deleted by the join operation
            }
            else
                // join to source
            {
                pxl2Vtx.at<int>(p) = GC_JNT_FGD; // join node to source (FG) -2=GC_JNT_FGD
                //fromSource = lambda; // edge deleted by the join operation
                //toSink = 0; // as toSink=0, weight of edge(source, sink) is unmodified
            }
            
            // Set n-weights and t-weights for non terminal neighbors
            // Update t-weights for terminal neighbors.
            int vtx = pxl2Vtx.at<int>(p);
            if (p.x > 0)
            {
                double w = leftW.at<double>(p);
                int n = pxl2Vtx.at<int>(Point(p.x - 1, p.y)); // equiv to at<int>(p.y, p.x-1)
                if (n >= 0)  // no terminal W-neighbor
                    if (vtx >= 0) // no terminal node
                        graph.addEdges(vtx, n, w, w);
                    else
                        graph.addTermWeights(n, (jfg(vtx) ? w : 0), (jbg(vtx) ? w : 0));
                    else
                        if (vtx >= 0)
                            graph.addTermWeights(vtx, (jfg(n) ? w : 0), (jbg(n) ? w : 0));
                        else
                            if (jbg(vtx) != jbg(n))
                                graph.sourceToSinkW += w;
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                int n = pxl2Vtx.at<int>(Point(p.x - 1, p.y - 1));
                if (n >= 0) // not terminal NW-neighbor
                    if (vtx >= 0) // not terminal node
                        graph.addEdges(vtx, n, w, w);
                    else
                        graph.addTermWeights(n, (jfg(vtx) ? w : 0), (jbg(vtx) ? w : 0));
                    else // neighbor is terminal
                        if (vtx >= 0)
                            graph.addTermWeights(vtx, (jfg(n) ? w : 0), (jbg(n) ? w : 0));
                        else
                            if (jbg(vtx) != jbg(n))
                                graph.sourceToSinkW += w;
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                int n = pxl2Vtx.at<int>(Point(p.x, p.y - 1));
                if (n >= 0)
                    if (vtx >= 0)
                        graph.addEdges(vtx, n, w, w);
                    else
                        graph.addTermWeights(n, (jfg(vtx) ? w : 0), (jbg(vtx) ? w : 0));
                    else
                        if (vtx >= 0)
                            graph.addTermWeights(vtx, (jfg(n) ? w : 0), (jbg(n) ? w : 0));
                        else
                            if (jbg(vtx) != jbg(n))
                                graph.sourceToSinkW += w;
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                int n = pxl2Vtx.at<int>(Point(p.x + 1, p.y - 1));
                if (n >= 0)
                    if (vtx >= 0)
                        graph.addEdges(vtx, n, w, w);
                    else
                        graph.addTermWeights(n, (jfg(vtx) ? w : 0), (jbg(vtx) ? w : 0));
                    else
                        if (vtx >= 0)
                            graph.addTermWeights(vtx, (jfg(n) ? w : 0), (jbg(n) ? w : 0));
                        else
                            if (jbg(vtx) != jbg(n))
                                graph.sourceToSinkW += w;
            }
        }
    }
}

/*
 Multithreaded estimateSegmentation with reduced graph
 */
static double estimateSegmentation_slim( GCGraph<double>& graph, Mat& mask, const Mat& ptx2Vtx )
{
    double flow = 0;
    
    int n_thread = std::thread::hardware_concurrency();
    
    double result[r_count];
    //double * resultPtr = &result[0];
    
    // launch parallel partial max flow computations
    current_region = 0;
    std::vector<std::thread> pool;
    
    for (int j = 0; j < n_thread; j++)
        pool.push_back(std::thread(worker, &graph, &result[0], 0));
    
    for (auto& t : pool)
        t.join();
    
    for (int i = 0; i < r_count; i++)
        flow += result[i];
    
    pool.clear();
    current_region = 0;
    
    for (int j = 0; j < n_thread; j++)
        pool.push_back(std::thread(worker, &graph, &result[0], 1));
    
    for (auto& t : pool)
        t.join();
    
    for (int i = 0; i < r_count; i++)
        flow += result[i];
    
    pool.clear();
    current_region = 0;
    
    // last call using the whole residual graph
    // flow +=graph.maxFlow();
    
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                int v = ptx2Vtx.at<int>(p);
                if (v == GC_JNT_BGD )
                {
                    mask.at<uchar>(p) = GC_PR_BGD;
                }
                else if (v == GC_JNT_FGD )
                {
                    mask.at<uchar>(p) = GC_PR_FGD;
                }
                else if (graph.inSourceSegment(v))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
    return flow;
}

/*
 Construct non reduced GCGraph.
 Vertices are indexed by the region number, for parallel computation of max flow.
 */
static void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                             const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                             GCGraph<double>& graph)
{
    int vtxCount = img.cols*img.rows,
    edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);
    
    // region numbering
    int h_size = (img.cols + TRANS) / r_split+1, v_size = (img.rows + TRANS) / r_split+1;
    
    std::vector<std::vector<int>> r_index(r_split, std::vector<int>(r_split));
    
    //quadtree(QT_HEIGHT, r_index);
    for (int i = 0; i < r_split; i++)
        for (int j = 0; j < r_split; j++)
            r_index[i][j] = i*r_split + j;
    
    graph.create(vtxCount, edgeCount);
    Point p;
    //int count = 0;
    for (p.y = 0; p.y < img.rows; p.y++)
    {
        for (p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int r = r_index[p.y / v_size][p.x / h_size];
            int r1 = r_index[p.y / v_size][(p.x + TRANS) / h_size];
            int r2 = r_index[(p.y + TRANS) / v_size][p.x / h_size];
            int r3 = r_index[(p.y + TRANS) / v_size][(p.x + TRANS) / h_size];
            
            int alt_r = r;
            
            if (r != r1)
                if (r == r2)
                    alt_r = r1;
                else
                    alt_r = r3;
                else
                    if (r != r2)
                        alt_r = r2;
            
            int vtxIdx = graph.addVtx(r, alt_r);
            
            Vec3b color = img.at<Vec3b>(p);
            
            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
            {
                fromSource = -log(bgdGMM(color));
                toSink = -log(fgdGMM(color));
            }
            else if (mask.at<uchar>(p) == GC_BGD)
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);
            
            // set n-weights
            if (p.x>0)
            {
                double w = leftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x>0 && p.y>0)
            {
                double w = upleftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y>0)
            {
                double w = upW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x<img.cols - 1 && p.y>0)
            {
                double w = uprightW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
        }
    }
}

/*
 Multithreaded estimateSegmentation with non reduced graph
 */
static double estimateSegmentation_threaded(GCGraph<double>& graph, Mat& mask)
{
    double flow = 0;
    current_region = 0;
    
    double result[r_count];
    std::vector<std::thread> pool;
    
    int n_thread = std::thread::hardware_concurrency();
    
    // launch parallel computations of partial max flows
    for (int lv = 0; lv < 1; lv++)
    {
        for (int j = 0; j < n_thread; j++)
            pool.push_back(std::thread(worker, &graph, &result[0], 0));
        
        for (auto& t : pool)
            t.join();
        
        for (int i = 0; (i << lv)< r_count; i++)
            flow += result[i << lv];
        
        pool.clear();
        current_region = 0;
    }
    
    // last call on the whole residual graph
    flow +=graph.maxFlow();
    
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++)
    {
        for (p.x = 0; p.x < mask.cols; p.x++)
        {
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
            {
                if (graph.inSourceSegment(p.y*mask.cols + p.x /*vertex index*/))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
    return flow;
}

/*
 Estimate segmentation using MaxFlow algorithm
 */
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                mask.at<uchar>(p) = GC_PR_FGD;
                else
                mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

void removeFluctuation(Mat& mask, Point starting_point){
    std::vector<Point> pixels_in_component;
    std::vector<Point> neighbours;
    
    neighbours.push_back(starting_point);
    uchar mask_val;
    Point neighbor_point;
    bool already_added;
    Mat res;
    mask.copyTo(res);
    res.setTo(GC_PR_BGD);
    
    while(!neighbours.empty()){
        already_added = false;
        Point p = neighbours.back();
        neighbours.pop_back();
        
        if (res.at<uchar>(p.y, p.x) == GC_FGD)
            already_added = true;
        
        if (!already_added) {
            pixels_in_component.push_back(p);
            res.at<uchar>(p.y, p.x) = GC_FGD;
            
            double x = p.x;
            double y = p.y;
            
            // left
            if (x > 0) {
                mask_val = mask.at<uchar>(y,x-1);
                neighbor_point = Point(x-1,y);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // right
            if (x < mask.cols-1) {
                mask_val = mask.at<uchar>(y,x+1);
                neighbor_point = Point(x+1,y);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // up
            if (y > 0) {
                mask_val = mask.at<uchar>(y-1,x);
                neighbor_point = Point(x,y-1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // down
            if (y < mask.rows-1) {
                mask_val = mask.at<uchar>(y+1,x);
                neighbor_point = Point(x,y+1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // up left
            if (x > 0 && y > 0) {
                mask_val = mask.at<uchar>(y-1,x-1);
                neighbor_point = Point(x-1,y-1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // up right
            if (x < mask.cols-1 && y > 0) {
                mask_val = mask.at<uchar>(y-1,x+1);
                neighbor_point = Point(x+1,y-1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // down left
            if (x > 0 && y < mask.rows-1) {
                mask_val = mask.at<uchar>(y+1,x-1);
                neighbor_point = Point(x-1,y+1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
            
            // down right
            if (x < mask.cols-1 && y < mask.rows-1) {
                mask_val = mask.at<uchar>(y+1,x+1);
                neighbor_point = Point(x+1,y+1);
                if(mask_val == GC_FGD || mask_val == GC_PR_FGD)
                    neighbours.push_back(neighbor_point);
            }
        }
    }
    res.copyTo(mask);
}

/*
 Multithreaded version of paintselection
 Non reduced graph
 */
void cv::paintselection(InputArray _img, InputOutputArray _mask,OutputArrayOfArrays contour,
                 InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                 int iterCount, int mode)
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    clock_t tStart, tEnd;
    
    //Downsampling
    /*
     https://stackoverflow.com/a/41148555/4582711
     cv::resize with INTER_NEAREST
     
     Input:
     [  1,   2,   3;
     4,   5,   6;
     7,   8,   9]

     Resized:
     [  1,   1,   2,   2,   3,   3;
     1,   1,   2,   2,   3,   3;
     4,   4,   5,   5,   6,   6;
     4,   4,   5,   5,   6,   6;
     7,   7,   8,   8,   9,   9;
     7,   7,   8,   8,   9,   9]
     

     */
    double c_ratio = 0.0;
    double r_ratio = 0.0;
    double ratio = 0.0;
    bool donwSampled = false;
    int orig_rows = img.rows;
    int orig_cols = img.cols;
    if (img.cols > 800 || img.rows > 600){
        c_ratio = img.cols/800.0;
        r_ratio = img.rows/600.0;
        ratio = (c_ratio > r_ratio)?r_ratio:c_ratio;
        printf("downscaled by: %f\n", ratio);
        fflush(stdout);
        resize( img, img, Size( img.cols/ratio, img.rows/ratio ),0,0,INTER_NEAREST);
        resize( mask, mask, Size( mask.cols/ratio, mask.rows/ratio ),0,0,INTER_NEAREST);
        donwSampled = true;
    }
    
    //Searching for starting seed pixel
    bool found_seed = false;
    Point starting_point;
    CV_Assert( !mask.empty() );
    for(int i=0; i< mask.cols; ++i){
        for(int j=0; j<mask.rows; ++j){
            if(mask.at<uchar>(j,i) == GC_FGD){  // at=> (row, column)
                starting_point = Point(i,j);    // Point=> (column, row)
                break;
                found_seed = true;
            }
        }
        if(found_seed)
            break;
    }

    if (img.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");

    GMM bgdGMM(bgdModel,8);
    GMM fgdGMM(fgdModel,4);

    Mat compIdxs(img.size(), CV_32SC1);

    if (mode == GC_INIT_WITH_MASK)
    {
        checkMask(img, mask);
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }

    CV_Assert( iterCount >= 0 );

    if (mode == GC_EVAL)
        checkMask(img, mask);

    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calcBeta(img);

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

    for (int i = 0; i < iterCount; i++)
    {
        GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);

        tStart = clock();
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        tEnd = clock();
        printf("construcGCGraph: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);

        tStart = clock();
//        estimateSegmentation_threaded(graph, mask);
        estimateSegmentation(graph, mask);
        tEnd = clock();
        printf("estimateSegmentation: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    }
    tStart = clock();
    removeFluctuation(mask,starting_point);
    tEnd = clock();
    printf("removeFluctuation: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    
    //Upsampling
    if(donwSampled){
        resize( mask, mask, Size( orig_cols, orig_rows ),0,0,INTER_NEAREST);
    }
    
    //Finding contours
    tStart = clock();
    findContours((mask & 1)*255, contour, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );
    tEnd = clock();
    printf("findContours: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    
}

/*
 Multithreded version of paintselection
 Reduced graph
 */
void cv::paintselection_slim(InputArray _img, InputOutputArray _mask,OutputArrayOfArrays contour,
                      InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                      int iterCount, int mode)
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    
    //Downsampling
    /*
     https://stackoverflow.com/a/41148555/4582711
     cv::resize with INTER_NEAREST
     
     Input:
     [  1,   2,   3;
     4,   5,   6;
     7,   8,   9]
     
     Resized:
     [  1,   1,   2,   2,   3,   3;
     1,   1,   2,   2,   3,   3;
     4,   4,   5,   5,   6,   6;
     4,   4,   5,   5,   6,   6;
     7,   7,   8,   8,   9,   9;
     7,   7,   8,   8,   9,   9]
     
     
     */
    double c_ratio = 0.0;
    double r_ratio = 0.0;
    double ratio = 0.0;
    bool donwSampled = false;
    int orig_rows = img.rows;
    int orig_cols = img.cols;
    if (img.cols > 800 || img.rows > 600){
        c_ratio = img.cols/800.0;
        r_ratio = img.rows/600.0;
        ratio = (c_ratio > r_ratio)?r_ratio:c_ratio;
        printf("downscaled by: %f\n", ratio);
        fflush(stdout);
        resize( img, img, Size( img.cols/ratio, img.rows/ratio ),0,0,INTER_NEAREST);
        resize( mask, mask, Size( mask.cols/ratio, mask.rows/ratio ),0,0,INTER_NEAREST);
        donwSampled = true;
    }
    
    //Searching for starting seed pixel
    bool found_seed = false;
    Point starting_point;
    CV_Assert( !mask.empty() );
    for(int i=0; i< mask.cols; ++i){
        for(int j=0; j<mask.rows; ++j){
            if(mask.at<uchar>(j,i) == GC_FGD){  // at=> (row, column)
                starting_point = Point(i,j);    // Point=> (column, row)
                break;
                found_seed = true;
            }
        }
        if(found_seed)
            break;
    }
                
    if (img.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");
    
    GMM bgdGMM(bgdModel,8);
    GMM fgdGMM(fgdModel,4);
    Mat compIdxs(img.size(), CV_32SC1);
    //Mat pxl2Vtx(img.size(), CV_32SC1);   // pixel vertices
    Mat pxl2Vtx(img.size(), CV_32S);
    
    clock_t tStart, tEnd;
    tStart = clock();
    if (mode == GC_INIT_WITH_MASK)
    {
        checkMask(img, mask);
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }
    
    CV_Assert( iterCount >= 0 );
    
    if (mode == GC_EVAL)
        checkMask(img, mask);
    
    const double gamma = 50;
    const double lambda = 9 * gamma;
    
    const double beta = calcBeta(img);
    
    Mat leftW, upleftW, upW, uprightW, sigmaNW;
    
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
    
    for (int i = 0; i < iterCount; i++)
    {
        GCGraph<double> graph;
        Mat mask2 = mask.clone();
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        tEnd = clock();
        printf("**************GMM model: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        
        
        
        tStart = clock();
        constructGCGraph_slim(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph, pxl2Vtx);
        tEnd = clock();
        printf("*************construcGCGraph slim: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        
        double flow;
        
#define TEST_VERSION
        
#ifdef TEST_VERSION
        GCGraph<double> graph2, graph3;
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph2);
        tStart = clock();
        flow = graph2.maxFlow();
        tEnd = clock();
        printf("***************seq. test standard flow: %f seq maxFlow time %.2f\n", flow+graph2.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph3);
        //Mat mask2 = mask.clone();
        tStart = clock();
        flow = estimateSegmentation_threaded(graph3, mask2);
        tEnd = clock();
        printf("**************test standard flow: %f estimateSegmentation time %.2f\n", flow+graph3.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        for (int i = 0; i < img.cols; i++)
            for (int j = 0; j < img.rows; j++)
                mask2.at<uchar>(j, i) = mask2.at<uchar>(j, i) << 2;
#endif
        
        tStart = clock();
        flow = graph.maxFlow();
        tEnd = clock();
        printf("***************seq. slim flow: %f seq maxFlow slim time %.2f\n", flow+graph.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        GCGraph<double> graph4;
        constructGCGraph_slim(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph4, pxl2Vtx);
        tStart = clock();
        flow = estimateSegmentation_slim(graph4, mask, pxl2Vtx);
        tEnd = clock();
        printf("**************slim flow %f estimateSegmentation slim time %.2fs\n", flow+graph4.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        cv::bitwise_or(mask2, mask, mask);
    }
    fflush(stdout);
                
    //Upsampling
    if(donwSampled){
        resize( mask, mask, Size( orig_cols, orig_rows ),0,0,INTER_NEAREST);
    }
    
    //Finding contours
    tStart = clock();
    findContours((mask & 1)*255, contour, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );
    tEnd = clock();
    printf("findContours: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    
}

/*
 Multithreaded version of grabCut
 Non reduced graph
 */
void cv::grabCut(InputArray _img, InputOutputArray _mask,OutputArrayOfArrays contour, Rect rect,
                 InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                 int iterCount, int mode)
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    clock_t tStart, tEnd;
    
    //Downsampling
    /*
     https://stackoverflow.com/a/41148555/4582711
     cv::resize with INTER_NEAREST
     
     Input:
     [  1,   2,   3;
     4,   5,   6;
     7,   8,   9]
     
     Resized:
     [  1,   1,   2,   2,   3,   3;
     1,   1,   2,   2,   3,   3;
     4,   4,   5,   5,   6,   6;
     4,   4,   5,   5,   6,   6;
     7,   7,   8,   8,   9,   9;
     7,   7,   8,   8,   9,   9]
     
     
     */
    double c_ratio = 0.0;
    double r_ratio = 0.0;
    double ratio = 0.0;
    bool donwSampled = false;
    int orig_rows = img.rows;
    int orig_cols = img.cols;
    if (img.cols > 800 || img.rows > 600){
        c_ratio = img.cols/800.0;
        r_ratio = img.rows/600.0;
        ratio = (c_ratio > r_ratio)?r_ratio:c_ratio;
        printf("downscaled by: %f\n", ratio);
        fflush(stdout);
        resize( img, img, Size( img.cols/ratio, img.rows/ratio ),0,0,INTER_NEAREST);
        resize( mask, mask, Size( mask.cols/ratio, mask.rows/ratio ),0,0,INTER_NEAREST);
        donwSampled = true;
    }
    
    if (img.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");
    
    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
    Mat compIdxs(img.size(), CV_32SC1);
    
    if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK)
    {
        if (mode == GC_INIT_WITH_RECT)
            initMaskWithRect(mask, img.size(), rect);
        else // flag == GC_INIT_WITH_MASK
            checkMask(img, mask);
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }
    
    CV_Assert( iterCount >= 0 );
    
    if (mode == GC_EVAL)
        checkMask(img, mask);
    
    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calcBeta(img);
    
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
    
    for (int i = 0; i < iterCount; i++)
    {
        GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        
        tStart = clock();
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        tEnd = clock();
        printf("construcGCGraph: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        
        tStart = clock();
        estimateSegmentation(graph, mask);
        tEnd = clock();
        printf("estimateSegmentation: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    }
    
    //Upsampling
    if(donwSampled){
        resize( mask, mask, Size( orig_cols, orig_rows ),0,0,INTER_NEAREST);
    }
    
    //Finding contours
    tStart = clock();
    findContours((mask & 1)*255, contour, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );
    tEnd = clock();
    printf("findContours: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    
}

/*
 Multithreded version of grabCut
 Reduced graph
 */
void cv::grabCut_slim(InputArray _img, InputOutputArray _mask, OutputArrayOfArrays contour, Rect rect,
                      InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                      int iterCount, int mode)
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();
    
    //Downsampling
    /*
     https://stackoverflow.com/a/41148555/4582711
     cv::resize with INTER_NEAREST
     
     Input:
     [  1,   2,   3;
     4,   5,   6;
     7,   8,   9]
     
     Resized:
     [  1,   1,   2,   2,   3,   3;
     1,   1,   2,   2,   3,   3;
     4,   4,   5,   5,   6,   6;
     4,   4,   5,   5,   6,   6;
     7,   7,   8,   8,   9,   9;
     7,   7,   8,   8,   9,   9]
     
     
     */
    double c_ratio = 0.0;
    double r_ratio = 0.0;
    double ratio = 0.0;
    bool donwSampled = false;
    int orig_rows = img.rows;
    int orig_cols = img.cols;
    if (img.cols > 800 || img.rows > 600){
        c_ratio = img.cols/800.0;
        r_ratio = img.rows/600.0;
        ratio = (c_ratio > r_ratio)?r_ratio:c_ratio;
        printf("downscaled by: %f\n", ratio);
        fflush(stdout);
        resize( img, img, Size( img.cols/ratio, img.rows/ratio ),0,0,INTER_NEAREST);
        resize( mask, mask, Size( mask.cols/ratio, mask.rows/ratio ),0,0,INTER_NEAREST);
        donwSampled = true;
    }
    
    //Searching for starting seed pixel
    bool found_seed = false;
    Point starting_point;
    CV_Assert( !mask.empty() );
    for(int i=0; i< mask.cols; ++i){
        for(int j=0; j<mask.rows; ++j){
            if(mask.at<uchar>(j,i) == GC_FGD){  // at=> (row, column)
                starting_point = Point(i,j);    // Point=> (column, row)
                break;
                found_seed = true;
            }
        }
        if(found_seed)
            break;
    }
    
    if (img.empty())
        CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");
    
    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
    Mat compIdxs(img.size(), CV_32SC1);
    //Mat pxl2Vtx(img.size(), CV_32SC1);   // pixel vertices
    Mat pxl2Vtx(img.size(), CV_32S);
    
    clock_t tStart, tEnd;
    tStart = clock();
    if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK)
    {
        if (mode == GC_INIT_WITH_RECT)
            initMaskWithRect(mask, img.size(), rect);
        else // flag == GC_INIT_WITH_MASK
            checkMask(img, mask);
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }
    
    CV_Assert( iterCount >= 0 );
    
    if (mode == GC_EVAL)
        checkMask(img, mask);
    
    const double gamma = 50;
    const double lambda = 9 * gamma;
    
    const double beta = calcBeta(img);
    
    Mat leftW, upleftW, upW, uprightW, sigmaNW;
    
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
    
    for (int i = 0; i < iterCount; i++)
    {
        GCGraph<double> graph;
        Mat mask2 = mask.clone();
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        tEnd = clock();
        printf("**************GMM model: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        
        
        
        tStart = clock();
        constructGCGraph_slim(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph, pxl2Vtx);
        tEnd = clock();
        printf("*************construcGCGraph slim: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        
        double flow;
        
#define TEST_VERSION
        
#ifdef TEST_VERSION
        GCGraph<double> graph2, graph3;
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph2);
        tStart = clock();
        flow = graph2.maxFlow();
        tEnd = clock();
        printf("***************seq. test standard flow: %f seq maxFlow time %.2f\n", flow+graph2.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph3);
        //Mat mask2 = mask.clone();
        tStart = clock();
        flow = estimateSegmentation_threaded(graph3, mask2);
        tEnd = clock();
        printf("**************test standard flow: %f estimateSegmentation time %.2f\n", flow+graph3.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        for (int i = 0; i < img.cols; i++)
            for (int j = 0; j < img.rows; j++)
                mask2.at<uchar>(j, i) = mask2.at<uchar>(j, i) << 2;
#endif
        
        tStart = clock();
        flow = graph.maxFlow();
        tEnd = clock();
        printf("***************seq. slim flow: %f seq maxFlow slim time %.2f\n", flow+graph.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        GCGraph<double> graph4;
        constructGCGraph_slim(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph4, pxl2Vtx);
        tStart = clock();
        flow = estimateSegmentation_slim(graph4, mask, pxl2Vtx);
        tEnd = clock();
        printf("**************slim flow %f estimateSegmentation slim time %.2fs\n", flow+graph4.sourceToSinkW, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
        cv::bitwise_or(mask2, mask, mask);
    }
    fflush(stdout);
    
    //Upsampling
    if(donwSampled){
        resize( mask, mask, Size( orig_cols, orig_rows ),0,0,INTER_NEAREST);
    }
    
    //Finding contours
    tStart = clock();
    findContours((mask & 1)*255, contour, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );
    tEnd = clock();
    printf("findContours: %.2fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
    
}
