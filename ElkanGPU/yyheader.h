#pragma once
#define DTYPE double
#define BLOCKSIZE 256
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