#ifndef ILQGAMES_UTILS_BIANRY_SAVE_H
#define ILQGAMES_UTILS_BIANRY_SAVE_H

#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

namespace ilqgames {


void saveBinary(const std::vector<Eigen::VectorXd>& vectors, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary); // Open in binary mode
    if (!file.is_open()) return;

    // Write the number of vectors (header)
    size_t count = vectors.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& vec : vectors) {
        // Write the size of vector
        Eigen::Index size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));

        // Write the raw data
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));
    }
    file.close();
    std::cout << "Saved binary to " << filename << std::endl;
}

std::vector<Eigen::VectorXd> loadBinary(const std::string& filename) {
    std::vector<Eigen::VectorXd> vectors;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return vectors;
    }

    // Read how many vectors are in the file
    size_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    vectors.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // Read the size of the vector
        Eigen::Index size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));

        // Prepare the vector and read raw data directly into it
        Eigen::VectorXd vec(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
        
        vectors.push_back(vec);
    }
    
    return vectors;
}

void saveToCSV(const std::vector<Eigen::VectorXd>& vectors, const std::string& filename) {
    std::ofstream file(filename);
    // Set precision to ensure you don't lose data
    file << std::setprecision(15); 

    for (const auto& vec : vectors) {
        for (int i = 0; i < vec.size(); ++i) {
            file << vec[i] << (i == vec.size() - 1 ? "" : ",");
        }
        file << "\n"; // Each vector on a new line
    }
    file.close();
}

}  // namespace ilqgames

#endif
