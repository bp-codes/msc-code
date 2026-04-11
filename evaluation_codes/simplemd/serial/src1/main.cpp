#ifndef MAIN_CPP
#define MAIN_CPP

#include "lib/SimpleMD/_simplemd.hpp"
#include "lib/Helper/_helper.hpp"




int main(int argc, char* argv[]) 
{
    // Read in path to configuration file     
    if (argc != 2) 
    {
        THROW_RUNTIME_ERROR("must give an input file e.g. ./SimpleMD.x input.json");
    }
    std::filesystem::path input_file = argv[1];


    SimpleMD::Run::run(input_file);
    return 0;
}



#endif

