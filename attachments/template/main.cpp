#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <cassert>
#include <limits>
#include <vulkan/vulkan_handles.hpp>
#include <fstream>
#include <utility>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	GLFWwindow *window = nullptr;
	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
	vk::raii::Device          device                  = nullptr;
	uint32_t                  queueIndex              = ~0;
	vk::raii::Queue           queue           = nullptr;
	vk::raii::SurfaceKHR      surface                 = nullptr;
	vk::raii::SwapchainKHR    swapChain               = nullptr;
	std::vector<vk::Image>    swapChainImages;
	vk::SurfaceFormatKHR      swapChainSurfaceFormat;
	vk::Extent2D              swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::PipelineLayout         pipelineLayout = nullptr;
	vk::raii::Pipeline               graphicsPipeline = nullptr;
	vk::raii::CommandPool            commandPool      = nullptr;
	std::vector<vk::raii::CommandBuffer>          commandBuffers;

	std::vector<vk::raii::Semaphore>              presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore>              renderFinishedSemaphores; // semaphores for swapchain operations (entirely on gpu)
	std::vector<vk::raii::Fence>                  drawFences; // waiting for previous frame to finish

	uint32_t frameIndex = 0;        // current frame, must be smaller than MAX_FRAMES_IN_FLIGHT

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createGraphicsPipeline();
		createCommandPool();
		createCommandBuffer();
		createSyncObjects();
	}

	void createCommandPool() {
		vk::CommandPoolCreateInfo poolInfo{.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		                                   .queueFamilyIndex = queueIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createCommandBuffer() {
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 2};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	void recordCommandBuffer(uint32_t imageIndex) {
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});
		transition_image_layout(imageIndex, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,
		    .clearValue  = clearColor};
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},
		    .layerCount           = 1,
		    .colorAttachmentCount = 1,
		    .pColorAttachments    = &attachmentInfo};
		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f)); // define the region the framebuffer will transform into
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent)); // filter out pixels outside of scissor region
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRendering();
		transition_image_layout(imageIndex, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR, vk::AccessFlagBits2::eColorAttachmentWrite, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eBottomOfPipe);
		commandBuffer.end();
	}

	void transition_image_layout(
	    uint32_t                imageIndex,
	    vk::ImageLayout         old_layout,
	    vk::ImageLayout         new_layout,
	    vk::AccessFlags2        src_access_mask,
	    vk::AccessFlags2        dst_access_mask,
	    vk::PipelineStageFlags2 src_stage_mask,
	    vk::PipelineStageFlags2 dst_stage_mask)
	{
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,
		    .srcAccessMask       = src_access_mask,
		    .dstStageMask        = dst_stage_mask,
		    .dstAccessMask       = dst_access_mask,
		    .oldLayout           = old_layout,
		    .newLayout           = new_layout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = swapChainImages[imageIndex],
		    .subresourceRange    = {
		           .aspectMask     = vk::ImageAspectFlagBits::eColor,
		           .baseMipLevel   = 0,
		           .levelCount     = 1,
		           .baseArrayLayer = 0,
		           .layerCount     = 1}};
		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},
		    .imageMemoryBarrierCount = 1,
		    .pImageMemoryBarriers    = &barrier};
		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	void createGraphicsPipeline() {
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo; // vertex data
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList}; // what kind of geometry will be drawn
		vk::PipelineViewportStateCreateInfo      viewportState{.viewportCount = 1, .scissorCount = 1};
		std::vector<vk::DynamicState>            dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor}; // dynamic state for properties that dont need to be baked into pipeline state
		vk::PipelineDynamicStateCreateInfo       dynamicState{
		          .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		          .pDynamicStates    = dynamicStates.data()
		};
		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False, 
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};
		vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};
		// alpha blending
		vk::PipelineColorBlendAttachmentState  colorBlendAttachment{
		     .blendEnable         = vk::True,
		     .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
		     .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		     .colorBlendOp        = vk::BlendOp::eAdd,
		     .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		     .dstAlphaBlendFactor = vk::BlendFactor::eZero,
		     .alphaBlendOp        = vk::BlendOp::eAdd,
		     .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
		vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};
		// uniform construction, can be updated at draw time
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 0, .pushConstantRangeCount = 0};
		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format};
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    {
				.stageCount          = 2,
				.pStages             = shaderStages,
				.pVertexInputState   = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState      = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState   = &multisampling,
				.pColorBlendState    = &colorBlending,
				.pDynamicState       = &dynamicState,
				.layout              = pipelineLayout,
				 .renderPass          = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainSurfaceFormat.format
			}
		};
		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule     shaderModule{device, createInfo};
		return shaderModule;
	}

	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();
		return buffer;
	}

	void createSwapChain() {
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent                                         = chooseSwapExtend(surfaceCapabilities);
		uint32_t                          minImageCount         = chooseSwapMinImageCount(surfaceCapabilities);
		std::vector<vk::SurfaceFormatKHR> availableFormats    = physicalDevice.getSurfaceFormatsKHR(*surface);
		swapChainSurfaceFormat                                  = chooseSwapSurfaceFormat(availableFormats);
		std::vector<vk::PresentModeKHR>   availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
		vk::SwapchainCreateInfoKHR      swapChainCreateInfo{.surface          = *surface,
		                                                    .minImageCount    = minImageCount,
		                                                    .imageFormat      = swapChainSurfaceFormat.format,
		                                                    .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
		                                                    .imageExtent      = swapChainExtent,
		                                                    .imageArrayLayers = 1,
		                                                    .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
		                                                    .imageSharingMode = vk::SharingMode::eExclusive,
		                                                    .preTransform     = surfaceCapabilities.currentTransform,
		                                                    .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		                                                    .presentMode      = chooseSwapPresentMode(availablePresentModes),
		                                                    .clipped          = true
		};
		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
		const auto formatIt = std::ranges::find_if(availableFormats, [](const auto &format) {
			return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		});
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes,
		                           [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :
		           vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtend(vk::SurfaceCapabilitiesKHR const& capabilities) {
		// resolution of swap chain images
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
		    std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
		};
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());
		vk::ImageViewCreateInfo imageViewCreateInfo{.viewType         = vk::ImageViewType::e2D,
		                                            .format           = swapChainSurfaceFormat.format,
		                                            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	void createSurface() {
		VkSurfaceKHR _surface; // C API surface
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	void pickPhysicalDevice() {
		auto physicalDevices = instance.enumeratePhysicalDevices();
		if (physicalDevices.empty())
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const &physicalDevice) { return isDeviceSuitable(physicalDevice); });
		if (devIter == physicalDevices.end())
			throw std::runtime_error("failed to find a suitable GPU!");
		physicalDevice = *devIter;
	}

	void createLogicalDevice() {
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// get the first index into queueFamilyProperties which supports both graphics and present
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				// found a queue family that supports both graphics and present
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}

		// query for Vulkan 1.3 features
		vk::StructureChain<vk::PhysicalDeviceFeatures2,
		                   vk::PhysicalDeviceVulkan11Features,
		                   vk::PhysicalDeviceVulkan13Features,
		                   vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
		    featureChain = {
		        {},                                    // vk::PhysicalDeviceFeatures2
		        {.shaderDrawParameters = true},        // vk::PhysicalDeviceVulkan11Features
		        {.synchronization2 = true, .dynamicRendering = true},            // vk::PhysicalDeviceVulkan13Features
		        {.extendedDynamicState = true}         // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
		    };

		// create a Device
		float                     queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{.pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		                                           .queueCreateInfoCount    = 1,
		                                           .pQueueCreateInfos       = &deviceQueueCreateInfo,
		                                           .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		                                           .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);
	}

	bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
	{
		// Check if the physicalDevice supports the Vulkan 1.3 API version
		bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

		// Check if any of the queue families support graphics operations
		auto queueFamilies    = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

		// Check if all required physicalDevice extensions are available
		auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions =
		    std::ranges::all_of(requiredDeviceExtension,
		                        [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
			                        return std::ranges::any_of(availableDeviceExtensions,
			                                                   [requiredDeviceExtension](auto const &availableDeviceExtension) { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
		                        });

		// Check if the physicalDevice supports the required features (shader draw parameters, dynamic rendering and extended dynamic state)
		auto features                 = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
		                                                                     vk::PhysicalDeviceVulkan11Features,
		                                                                     vk::PhysicalDeviceVulkan13Features,
		                                                                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
		bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
		                                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		                                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

		// Return true if the physicalDevice meets all the criteria
		return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
	}

	void createInstance() 
	{
		// get required layers
		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		auto layerProperties = context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](auto const &requiredLayer) {
			return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
		});
		// check if required layers are supported
		if (unsupportedLayerIt != requiredLayers.end())
		{
			throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
		}

		constexpr vk::ApplicationInfo appInfo{
		    .pApplicationName   = "Hello Triangle",
		    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		    .pEngineName        = "No Engine",
		    .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		    .apiVersion         = vk::ApiVersion14
		};

		auto requiredExtensions  = getRequiredInstanceExtensions();
		auto     extensionProperties = context.enumerateInstanceExtensionProperties();
		// check if required extensions are supported
		auto unsupportedPropertyIt = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const &requiredExtension) {
			return std::ranges::none_of(extensionProperties, [requiredExtension](auto const &extensionProperty) {
				return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
			});
		});

		if (unsupportedPropertyIt != requiredExtensions.end())
		{
			throw std::runtime_error("Required extension not supported " + std::string(*unsupportedPropertyIt));
		}

		std::cout << "available extensions:\n";
		for (const auto &extension : extensionProperties)
		{
			std::cout << '\t' << extension.extensionName << '\n';
		}

		vk::InstanceCreateInfo createInfo{
		    .pApplicationInfo        = &appInfo,
		    .enabledLayerCount                              = static_cast<uint32_t>(requiredLayers.size()),
		    .ppEnabledLayerNames                              = requiredLayers.data(),
		    .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
		    .ppEnabledExtensionNames = requiredExtensions.data()
		};
		instance = vk::raii::Instance(context, createInfo);
	}

	std::vector<const char*> getRequiredInstanceExtensions() {
		uint32_t glfwExtensionCount = 0;
		auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		return extensions;
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		device.waitIdle();
	}

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void drawFrame() {
		// wait for previous frame
		auto fenceResult = device.waitForFences(*drawFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}
		device.resetFences(*drawFences[frameIndex]);
		// acquire next image
		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);
		recordCommandBuffer(imageIndex);
		// submit command buffer
		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{.waitSemaphoreCount   = 1,
		                                  .pWaitSemaphores      = &*presentCompleteSemaphores[frameIndex],
		                                  .pWaitDstStageMask    = &waitDestinationStageMask,
		                                  .commandBufferCount   = 1,
		                                  .pCommandBuffers      = &*commandBuffers[frameIndex],
		                                  .signalSemaphoreCount = 1,
		                                  .pSignalSemaphores    = &*renderFinishedSemaphores[frameIndex]};
		queue.submit(submitInfo, *drawFences[frameIndex]);
		const vk::PresentInfoKHR presentInfoKHR{
		    .waitSemaphoreCount = 1,
		    .pWaitSemaphores    = &*renderFinishedSemaphores[frameIndex],
		    .swapchainCount     = 1,
		    .pSwapchains        = &*swapChain,
		    .pImageIndices      = &imageIndex};
		result = queue.presentKHR(presentInfoKHR);
		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void createSyncObjects() {
		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			drawFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
