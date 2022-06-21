#pragma once

#include "vulkan/vulkan.h"
#include "SDL.h"
#include <vector>
#include <glm/glm.hpp>
#include <array>

#define DEBUG_MODE 1

#define MAX_VK_EXTENSIONS 10
#define MAX_VK_EXTENSION_NAME_SIZE 256

#define MAX_FRAMES_IN_FLIGHT 2

class VulkanApp {

public:
	void run();

private:

	struct UniformBufferObject {
		alignas(16) glm::mat4 model;
		alignas(16) glm::mat4 view;
		alignas(16) glm::mat4 proj;
	};

	struct SwapChainDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	struct QueueFamily {
		uint32_t index;
		bool exists = false;
	};

	struct QueueFamilies {
		QueueFamily graphicsFamily;
		QueueFamily presentFamily;

		bool isComplete() {
			return graphicsFamily.exists && presentFamily.exists;
		}
	};

	struct Vertex {
		glm::vec2 pos;
		glm::vec3 color;

		static VkVertexInputBindingDescription GetBindingDescription() {
			VkVertexInputBindingDescription description{};
			description.binding = 0;
			description.stride = sizeof(Vertex);
			description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			return description;
		}

		static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions() {
			std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);

			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset = offsetof(Vertex, color);

			return attributeDescriptions;
		}
	};

	// const int MAX_FRAMES_IN_FLIGHT = 2;

	void init();
	void mainLoop();
	void cleanup();

	void updateUniformBuffer(uint32_t currentFrame);
	void recreateSwapChain();
	void drawFrame();
	void createInstance();
	void createSurface();
	bool checkValidationSupport();
	void createSwapChain();
	void getNecessaryInstanceExtensions(char** extensionNamesBuffer, int* numExtensions);
	VkResult setupDebugCallbacks();
	VkResult destroyCallbacks();
	void populateCreateDebugUtilsMessenger(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	void pickPhysicalDevice();
	void createLogicalDevice();
	QueueFamilies getQueueFamiliesForPhysDevice(const VkPhysicalDevice& device);
	bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
	SwapChainDetails querySwapChainSupport(const VkPhysicalDevice& device);
	bool isPhysDevSuitable(const VkPhysicalDevice& device);
	VkSurfaceFormatKHR chooseSwapChainFormat(const std::vector<VkSurfaceFormatKHR>& formats);
	VkPresentModeKHR chooseSwapChainPresentMode(const std::vector<VkPresentModeKHR>& presentModes);
	VkExtent2D chooseSwapChainExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	void createImageViews();
	void createGraphicsPipeline();
	void createRenderPass();
	VkShaderModule createShaderModule(const std::vector<char>& code);
	void createFrameBuffers();
	void createCommandPool();
	void createCommandBuffers();
	void writeToCommandBuffer(VkCommandBuffer& commandBuffer, uint32_t imageIdx);
	void createSyncObjects();
	void cleanUpSwapChain();
	void createIndexBuffer();
	void createVertexBuffer();
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	void createDescriptorSetLayout();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();

	const char* validationLayers[1] = { "VK_LAYER_KHRONOS_validation" };
	int numValidationLayers = 1;
	const char* extraNecessaryInstanceExts[1] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
	int numNecessaryInstanceExts = 1;
	const char* extraNecessaryDevExtensions[1] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	int numNecessaryDevExts = 1;

#if DEBUG_MODE == 1
	bool includeValidation = true;
#else
	bool includeValidation = false;
#endif

	SDL_Window* window;
	int windowWidth = 600;
	int windowHeight = 600;
	bool running = true;
	VkInstance vkInstance = VK_NULL_HANDLE;
	VkSurfaceKHR vkSurface;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages = {};
	std::vector<VkImageView> swapChainImageViews = {};
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkPipelineLayout pipelineLayout;
	VkRenderPass renderPass;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFrameBuffers;
	VkCommandPool graphicsCommandPool;
	std::vector<VkCommandBuffer> graphicsCommandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	VkDescriptorSetLayout descriptorSetLayout;
	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	VkDescriptorPool descriptorPool;

	uint32_t currentFrame = 0;

	const std::vector<Vertex> vertices = {
		{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
		{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
	};

	const std::vector<uint16_t> indices = {
		0, 1, 2, 2, 3, 0
	};
};
