/**
 * @author Ethan Emmons Ian McVann Luke Langefels
 * Primary cpp file for vision
 */

#include <json.hpp>
#include <iostream>
#include <fstream>

using namespace nlohmann;

int main() {
    std::ifstream valuesFile("values.json");
    json values;

    valuesFile >> values;
}