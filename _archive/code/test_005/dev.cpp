#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>
#include <sycl/sycl.hpp>

struct Person {
    std::string first_name;
    std::string last_name;
    int age;
};

struct SyclPerson {
    char first_name[16];
    char last_name[16];
    int age;

    bool operator<(const SyclPerson& other) const {
        int lname_cmp = std::strcmp(last_name, other.last_name);
        if (lname_cmp != 0) return lname_cmp < 0;

        int fname_cmp = std::strcmp(first_name, other.first_name);
        if (fname_cmp != 0) return fname_cmp < 0;

        return age < other.age;
    }
};


// Convert C++ Person → SYCL-friendly version
std::vector<SyclPerson> copy_to_sycl_people(const std::vector<Person>& people) {
    std::vector<SyclPerson> result;
    result.reserve(people.size());
    for (const auto& p : people) {
        SyclPerson sp{};
        std::strncpy(sp.first_name, p.first_name.c_str(), sizeof(sp.first_name) - 1);
        std::strncpy(sp.last_name, p.last_name.c_str(), sizeof(sp.last_name) - 1);
        sp.age = p.age;
        result.push_back(sp);
    }
    return result;
}

// Convert SYCL version → back to standard C++ Person
std::vector<Person> copy_from_sycl_people(const std::vector<SyclPerson>& sycl_people) {
    std::vector<Person> result;
    result.reserve(sycl_people.size());
    for (const auto& sp : sycl_people) {
        result.push_back({
            std::string(sp.first_name),
            std::string(sp.last_name),
            sp.age
        });
    }
    return result;
}


// Comparison function for sorting
bool compare_people(const Person& a, const Person& b) {
    if (a.last_name != b.last_name) return a.last_name < b.last_name;
    if (a.first_name != b.first_name) return a.first_name < b.first_name;
    return a.age < b.age;
}

// Generate a random Person
Person generate_random_person(std::mt19937& rng, 
                              const std::vector<std::string>& first_names,
                              const std::vector<std::string>& last_names) {
    std::uniform_int_distribution<int> age_dist(1, 100);
    std::uniform_int_distribution<size_t> first_dist(0, first_names.size() - 1);
    std::uniform_int_distribution<size_t> last_dist(0, last_names.size() - 1);

    return {
        first_names[first_dist(rng)],
        last_names[last_dist(rng)],
        age_dist(rng)
    };
}

bool compare(const Person& a, const Person& b) {
    if (a.last_name != b.last_name) return a.last_name < b.last_name;
    if (a.first_name != b.first_name) return a.first_name < b.first_name;
    return a.age < b.age;
}

// Merge two sorted halves
void merge(std::vector<Person>& data, int left, int mid, int right) {
    std::vector<Person> temp;
    int i = left, j = mid;

    while (i < mid && j < right) {
        if (compare(data[i], data[j])) temp.push_back(data[i++]);
        else temp.push_back(data[j++]);
    }

    while (i < mid) temp.push_back(data[i++]);
    while (j < right) temp.push_back(data[j++]);

    std::copy(temp.begin(), temp.end(), data.begin() + left);
}

void merge_sort(std::vector<Person>& data, int left, int right) {
    if (right - left <= 1) return;  // Base case: 0 or 1 element

    int mid = left + (right - left) / 2;
    merge_sort(data, left, mid);
    merge_sort(data, mid, right);
    merge(data, left, mid, right);
}


void merge_sort_parallel(std::vector<Person>& data, int left, int right, int depth = 0) {
    if (right - left <= 1000) {  // Base case threshold
        std::sort(data.begin() + left, data.begin() + right, compare);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(data) if(depth < 4)
    merge_sort_parallel(data, left, mid, depth + 1);

    #pragma omp task shared(data) if(depth < 4)
    merge_sort_parallel(data, mid, right, depth + 1);

    #pragma omp taskwait
    merge(data, left, mid, right);
}

// Serial merge sort (called in SYCL kernel)
void merge_sort_iterative(SyclPerson* data, SyclPerson* temp, int size) {
    for (int width = 1; width < size; width *= 2) {
        for (int i = 0; i < size; i += 2 * width) {
            int left = i;
            int mid = std::min(i + width, size);
            int right = std::min(i + 2 * width, size);

            int idx = left, a = left, b = mid;
            while (a < mid && b < right) {
                if (data[a] < data[b]) {
                    temp[idx++] = data[a++];
                } else {
                    temp[idx++] = data[b++];
                }
            }
            while (a < mid) temp[idx++] = data[a++];
            while (b < right) temp[idx++] = data[b++];
        }
        // Copy temp back to data
        for (int i = 0; i < size; ++i) {
            data[i] = temp[i];
        }
    }
}

bool compare_person(const Person& a, const Person& b) {
    if (a.last_name != b.last_name) return a.last_name < b.last_name;
    if (a.first_name != b.first_name) return a.first_name < b.first_name;
    return a.age < b.age;
}

int main() {

    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> first_names = {
        "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah", "Ian", "Julia",
        "Kevin", "Laura", "Michael", "Nina", "Oliver", "Paula", "Quentin", "Rachel", "Steve", "Tina"
    };

    std::vector<std::string> last_names = {
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson",
        "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White"
    };

    constexpr size_t num_people = 8000000;
    std::vector<Person> people;
    people.reserve(num_people);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < num_people; ++i) {
        people.push_back(generate_random_person(rng, first_names, last_names));
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Setup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";


    auto sequential_people = people; 
    auto parallel_people = people; 

    start = std::chrono::high_resolution_clock::now();
    merge_sort(sequential_people, 0, people.size());
    end = std::chrono::high_resolution_clock::now();

    bool sorted = std::is_sorted(sequential_people.begin(), sequential_people.end(), compare_person);
    std::cout << "Sorted: " << sorted << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";


    start = std::chrono::high_resolution_clock::now();
    merge_sort_parallel(parallel_people, 0, people.size());
    end = std::chrono::high_resolution_clock::now();

    sorted = std::is_sorted(parallel_people.begin(), parallel_people.end(), compare_person);
    std::cout << "Sorted: " << sorted << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";



    // Step 2: Convert to SYCL-friendly format
    start = std::chrono::high_resolution_clock::now();
    std::vector<SyclPerson> sycl_people_host = copy_to_sycl_people(people);


    sycl::queue q;
    SyclPerson* sycl_people = sycl::malloc_shared<SyclPerson>(num_people, q);
    SyclPerson* temp = sycl::malloc_shared<SyclPerson>(num_people, q);
    std::copy(sycl_people_host.begin(), sycl_people_host.end(), sycl_people);

    // Run kernel with non-recursive sort
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            merge_sort_iterative(sycl_people, temp, static_cast<int>(num_people));
        });
    }).wait();

    // Copy back results and convert to Person
    std::vector<SyclPerson> sorted_sycl_people(sycl_people, sycl_people + num_people);
    std::vector<Person> sorted_people = copy_from_sycl_people(sorted_sycl_people);
    end = std::chrono::high_resolution_clock::now();
    
    sorted = std::is_sorted(sorted_people.begin(), sorted_people.end(), compare_person);
    std::cout << "Sorted: " << sorted << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";



    /*

    // Parallel sort
    auto sequential_people = people; // copy for sequential sort
    auto start = std::chrono::high_resolution_clock::now();

    
    std::sort(sequential_people.begin(), sequential_people.end(), compare_people);


    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential sort time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";


              
    */

              

    return 0;
}



// g++ -std=c++17 -O2 -pthread -ltbb dev.cpp -o dev.x