#pragma once
#define DTYPE double
#define BLOCKSIZE 256
#define _HUGE_ENUF  1e+300
#define INFINITY   ((float)(_HUGE_ENUF * _HUGE_ENUF))

//const int BLOCKSIZE = 256;

typedef struct PointInfo
{
    //Indices of old and new assigned centroids
    int centroidIndex;
    int oldCentroid;

    //The current upper bound
    DTYPE uprBound;
}point;

typedef struct CentInfo {
    //Centroid's group index
    int groupNum;

    //Centroid's drift after updating
    DTYPE drift;

    //number of data points assigned to centroid
    int count;
} cent;