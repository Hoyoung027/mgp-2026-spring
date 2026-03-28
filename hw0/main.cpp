#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

int main() {
    /*
    Write a program that reads a string from /data/hw0/data.txt and prints its reverse. 
    Do not hardcode, as the data will be changed.
    */
    std::ifstream file("/data/hw0/data.txt");
    std::string str;
    std::getline(file, str);

    std::reverse(str.begin(), str.end());
    std::cout << str << "\n";

    return 0;
}
