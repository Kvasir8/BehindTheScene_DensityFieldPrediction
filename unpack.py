import numpy as np


def unpack(compressed):
    """given a bit encoded voxel grid, make a normal voxel grid out of it."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def pack(uncompressed):
    uncompressed_r = uncompressed.reshape(-1, 8)
    compressed = uncompressed_r.dot(
        1 << np.arange(uncompressed_r.shape[-1] - 1, -1, -1)
    )
    return compressed


def main():
    voxel_grid = np.random.randint(0, 2, (32, 32, 4)).astype(np.float32)
    voxel_grid[...] = 0
    voxel_grid[..., :2] = 1
    input = voxel_grid.reshape(-1)
    compressed = pack(input).astype(np.uint8)
    np.save("voxel_grid.npy", compressed)
    loaded_compressed = np.load("voxel_grid.npy")
    # loaded_compressed = np.fromfile("voxel_grid.npy", dtype=np.uint8)
    uncompressed = unpack(loaded_compressed)
    output = uncompressed.reshape(voxel_grid.shape).astype(np.float32)
    pass


if __name__ == "__main__":
    main()
