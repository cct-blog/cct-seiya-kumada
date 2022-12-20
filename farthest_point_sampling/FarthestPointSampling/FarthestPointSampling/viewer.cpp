#include "viewer.h"

auto view_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) -> void {
    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    //pcl::io::loadPCDFile("pcd_data.pcd", *cloud);

    //pcl::visualization::CloudViewer viewer("Cloud Viewer");

    ////blocks until the cloud is actually rendered
    //viewer.showCloud(cloud);

    ////use the following functions to get access to the underlying more advanced/powerful
    ////PCLVisualizer

    ////This will only get called once
    //viewer.runOnVisualizationThreadOnce(viewerOneOff);

    ////This will get called once per visualization iteration
    //viewer.runOnVisualizationThread(viewerPsycho);
    //while (!viewer.wasStopped())
    //{
    //    //you can also do cool processing here
    //    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //    //and you should guard against race conditions yourself...
    //    user_data++;
    //}
}
