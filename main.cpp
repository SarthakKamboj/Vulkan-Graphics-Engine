#include <iostream>
#include <stdexcept>
#include "vulkanApp.h"


int main(int argc, char* args[]) {
	VulkanApp vulkanApp;
	try {
		vulkanApp.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return -1;
}
