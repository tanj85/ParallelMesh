#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <queue>

using namespace std;

using Grid = vector<vector<int>>;

Grid generate_random_grid(int n, int k, unsigned seed) {
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, k-1);
    Grid grid(n, vector<int>(n));
    for (auto& row : grid)
        for (auto& cell : row)
            cell = dist(rng);
    return grid;
}

Grid generate_blocky_grid(int n, int k, int patch_size, unsigned seed) {
    mt19937 rng(seed);
    uniform_int_distribution<int> color_dist(0, k-1);
    Grid grid(n, vector<int>(n, 0));
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    for (int i = 0; i < n * n / patch_size; ++i) {
        int color = color_dist(rng);
        int x = rng() % n, y = rng() % n;
        queue<pair<int, int>> q;
        q.push({x, y});
        int count = 0;

        while (!q.empty() && count < patch_size) {
            auto [cx, cy] = q.front(); q.pop();
            if (cx < 0 || cx >= n || cy < 0 || cy >= n) continue;
            if (visited[cx][cy]) continue;

            visited[cx][cy] = true;
            grid[cx][cy] = color;
            count++;

            for (int d = 0; d < 4; ++d)
                q.push({cx + dx[d], cy + dy[d]});
        }
    }

    // Fill unvisited with random colors
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!visited[i][j])
                grid[i][j] = color_dist(rng);

    return grid;
}

void save_grid(const Grid& grid, const string& filename) {
    ofstream out(filename);
    for (const auto& row : grid) {
        for (int cell : row)
            out << cell << ' ';
        out << '\n';
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <n> <k> <mode: random|blocky> [seed] <output_file>\n";
        return 1;
    }

    int n = stoi(argv[1]);
    int k = stoi(argv[2]);
    string mode = argv[3];
    unsigned seed;
    string output_file;

    if (argc == 6) {
        seed = static_cast<unsigned>(stoi(argv[4]));
        output_file = argv[5];
    } else {
        seed = static_cast<unsigned>(time(nullptr));  // fallback to current time
        output_file = argv[4];
        cout << "Using default seed: " << seed << '\n';
    }

    Grid grid;
    if (mode == "random")
        grid = generate_random_grid(n, k, seed);
    else if (mode == "blocky")
        grid = generate_blocky_grid(n, k, n / 2, seed);  // block size n/2
    else {
        cerr << "Unknown mode: " << mode << '\n';
        return 1;
    }

    save_grid(grid, output_file);
    cout << "Grid saved to " << output_file << '\n';
    return 0;
}
