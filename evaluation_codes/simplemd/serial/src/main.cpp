#ifndef MAIN_CPP
#define MAIN_CPP

/*********************************************************************************************************************************/
#include <filesystem>   // std::filesystem::path
#include "lib/SimpleMD/_simplemd.hpp"
#include "lib/Helper/_helper.hpp"
/*********************************************************************************************************************************/


/**
 * @brief Entry point for the SimpleMD executable.
 *
 * Expects a single argument specifying the path to the JSON input configuration file.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 *
 * @return int Exit status code.
 *
 * @throws std::runtime_error Via THROW_RUNTIME_ERROR if input file is not provided.
 */
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        THROW_RUNTIME_ERROR("must give an input file e.g. ./SimpleMD.x input.json");
    }

    const std::filesystem::path input_file {argv[1]};

    SimpleMD::Run::run(input_file);

    return 0;
}

#endif
