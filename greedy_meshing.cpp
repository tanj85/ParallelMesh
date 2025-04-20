#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <omp.h>

struct Rect { int x, y, width, height, val; };

// 1) load flat grid + dims
std::vector<int> readGridFromFile(const std::string& filename,
                                  int& width, int& height)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<int> grid;
    width = height = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int v;
        while (iss >> v) row.push_back(v);
        if (row.empty()) continue;
        if (width == 0) width = (int)row.size();
        for (int x: row) grid.push_back(x);
        height++;
    }
    return grid;
}

// 2) greedyMeshChunk(appends into rects)
void greedyMeshChunk(const int* grid,
                     int totalWidth,
                     int x0, int y0,
                     int w, int h,
                     char* visited,
                     std::vector<Rect>& rects)
{
    rects.reserve(rects.size() + (w*h)/8);
    for (int dy = 0; dy < h; ++dy) {
        int gy = y0 + dy;
        for (int dx = 0; dx < w; ++dx) {
            int gx = x0 + dx;
            int idx = gy*totalWidth + gx;
            if (visited[idx] || grid[idx] == 0) continue;
            int val = grid[idx];

            // expand width
            int W = 0;
            while (dx+W < w) {
                int i2 = gy*totalWidth + (gx+W);
                if (grid[i2]!=val || visited[i2]) break;
                ++W;
            }

            // expand height
            int H = 1;
            bool ok;
            do {
                ok = true;
                if (gy+H >= y0+h) break;
                for (int xx = 0; xx < W; ++xx) {
                    int i2 = (gy+H)*totalWidth + (gx+xx);
                    if (grid[i2]!=val || visited[i2]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) ++H;
            } while (ok);

            // mark visited
            for (int yy = 0; yy < H; ++yy)
                for (int xx = 0; xx < W; ++xx)
                    visited[(gy+yy)*totalWidth + (gx+xx)] = 1;

            rects.push_back({gx, gy, W, H, val});
        }
    }
}

// 3) recursiveGreedyMesh with local leaf buffers + critical merge
void recursiveGreedyMesh(const int* grid,
                        int totalWidth, int totalHeight,
                        int x0, int y0, int w, int h,
                        int chunkSize,
                        std::vector<Rect>& output,
                        char* visited)
{
    if (w <= chunkSize && h <= chunkSize) {
        // leaf: do local, then merge under lock
        std::vector<Rect> local;
        greedyMeshChunk(grid, totalWidth, x0, y0, w, h, visited, local);
        #pragma omp critical
        output.insert(output.end(), local.begin(), local.end());
        return;
    }

    int midW = w/2, midH = h/2;
    int area = w*h;
    int threads = omp_get_max_threads();
    int totalCells = totalWidth * totalHeight;
    int min_grain = totalCells / (threads * 4);
    bool do_parallel = (area >= min_grain);

    if (do_parallel) {
        #pragma omp task shared(output, visited)
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0,      y0,      midW,   midH,
                            chunkSize, output, visited);

        #pragma omp task shared(output, visited)
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0+midW, y0,      w-midW, midH,
                            chunkSize, output, visited);

        #pragma omp task shared(output, visited)
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0,      y0+midH, midW,   h-midH,
                            chunkSize, output, visited);

        #pragma omp task shared(output, visited)
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0+midW, y0+midH, w-midW, h-midH,
                            chunkSize, output, visited);

        #pragma omp taskwait
    } else {
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0,      y0,      midW,   midH,
                            chunkSize, output, visited);
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0+midW, y0,      w-midW, midH,
                            chunkSize, output, visited);
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0,      y0+midH, midW,   h-midH,
                            chunkSize, output, visited);
        recursiveGreedyMesh(grid, totalWidth, totalHeight,
                            x0+midW, y0+midH, w-midW, h-midH,
                            chunkSize, output, visited);
    }
}

// 4) top‐level parallel wrapper
std::vector<Rect> recursiveParallelMeshing(const std::vector<int>& grid,
                                           int width, int height,
                                           int chunkSize)
{
    std::vector<Rect> result;
    std::vector<char> visited(width * height, 0);

    #pragma omp parallel
    {
        #pragma omp single nowait
        recursiveGreedyMesh(grid.data(),
                            width, height,
                            0, 0, width, height,
                            chunkSize,
                            result,
                            visited.data());
    }
    return result;
}

// --- updated main() ---
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./greedy_meshing <input_file>\n";
        return 1;
    }

    int width, height;
    auto grid = readGridFromFile(argv[1], width, height);

    std::cout << "Detected hardware threads: "
              << std::thread::hardware_concurrency() << "\n";
    std::cout << "OpenMP max threads: "
              << omp_get_max_threads() << "\n\n";

    // serial
    std::vector<char> vis0(width*height,0);
    std::vector<Rect> serial;
    auto t0 = std::chrono::high_resolution_clock::now();
    greedyMeshChunk(grid.data(), width, 0,0, width,height, vis0.data(), serial);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Serial runtime: "
              << std::chrono::duration<double>(t1-t0).count()
              << " s\n\n";

    // parallel
    int chunkSize = 5000;  // tune this
    t0 = std::chrono::high_resolution_clock::now();
    auto parallel = recursiveParallelMeshing(grid, width, height, chunkSize);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel runtime: "
              << std::chrono::duration<double>(t1-t0).count()
              << " s\n\n";

    std::cout << "Serial count:   " << serial.size()   << "\n";
    std::cout << "Parallel count: " << parallel.size() << "\n";
    return 0;
}
