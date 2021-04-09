from spconv.utils import VoxelGeneratorV2

kitti_voxel_generator = VoxelGeneratorV2(
    voxel_size=[0.05, 0.05, 0.1],
    point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    max_num_points=5,
    max_voxels=20000,
    full_mean=False,
    block_filtering=False,
    block_factor=0,
    block_size=0,
    height_threshold=0.0)
