# | Level | Identifier | Full Name | Parameter Range | CPU Usage Scenarios | GPU Usage Scenarios |
# |-------|------------|-----------|-----------------|---------------------|---------------------|
# | 0 | nano | Nano Model | < 1M | Ultra-low power devices (wearables, IoT sensors), real-time mobile apps with simple classification tasks, basic vision functions in embedded systems | Typically not needed, designed for CPU efficiency |
# | 1 | tiny | Tiny Model | 1M-3M | Mid-range mobile devices for real-time applications, edge computing devices, continuous monitoring on battery-powered devices | Batch processing on entry-level GPUs, low-latency edge AI systems |
# | 2 | small | Small Model | 3M-5M | Complex vision apps on high-end mobile devices, real-time processing on laptops, lightweight deployment on edge servers | Real-time inference on entry-level GPUs, moderate complexity tasks on mobile GPUs |
# | 3 | compact | Compact Model | 5M-10M | Medium complexity vision tasks on desktop CPUs, batch processing on server CPUs, complex applications on high-end edge devices | Efficient processing on entry to mid-range GPUs, complex vision tasks on mobile GPUs |
# | 4 | medium | Medium Model | 10M-25M | Complex vision tasks on high-performance CPUs, batch processing on multi-core servers, production environments with relaxed latency requirements | Real-time applications on mid-range GPUs, standard vision models for cloud services |
# | 5 | large | Large Model | 25M-50M | Batch processing on high-performance multi-core CPUs, server deployments with moderate latency requirements, compromise choice when resources are limited | Complex vision tasks on mid to high-end GPUs, standard production models in cloud services, medium-scale computer vision systems |
# | 6 | xlarge | Extra Large Model | 50M-100M | Offline processing on high-end server CPUs, batch processing scenarios without real-time requirements, production environments where latency is acceptable | High-precision vision tasks on mid to high-end GPUs, production systems requiring high accuracy, standard models in research environments |
# | 7 | xxlarge | Double Extra Large Model | 100M-250M | Only suitable for offline processing on multi-core high-performance servers, long inference times, requires large memory support | High-end GPU or multi-GPU systems, large-scale vision systems requiring high precision, enterprise-level AI solutions |
# | 8 | huge | Huge Model | 250M-500M | Not recommended for CPU inference (extremely inefficient), only for special offline analysis scenarios, requires substantial memory and computational resources | High-end GPU or multi-GPU systems, data center-level vision processing tasks, research and applications requiring extreme precision |
# | 9 | giant | Giant Model | 500M-1B | Not suitable for CPU inference, difficult to run efficiently even on high-end servers | Multi-GPU or GPU clusters, large-scale vision tasks in high-end data centers, cutting-edge research at top research institutions |
# | 10 | colossal | Colossal Model | >1B | Completely unsuitable for CPU inference | Requires multi-GPU clusters or specialized AI acceleration hardware, frontier research at large research institutions, special application scenarios with extremely high resource requirements |

def classify_model_by_params(params_in_millions):
    """
    Classify a model based on its parameter count.

    Args:
        params_in_millions: Number of parameters in millions (float)

    Returns:
        level: Integer level from 0-10
        identifier: String identifier for the model size category
    """
    if params_in_millions < 1:
        return 0, "nano"
    elif params_in_millions < 3:
        return 1, "tiny"
    elif params_in_millions < 5:
        return 2, "small"
    elif params_in_millions < 10:
        return 3, "compact"
    elif params_in_millions < 25:
        return 4, "medium"
    elif params_in_millions < 50:
        return 5, "large"
    elif params_in_millions < 100:
        return 6, "xlarge"
    elif params_in_millions < 250:
        return 7, "xxlarge"
    elif params_in_millions < 500:
        return 8, "huge"
    elif params_in_millions < 1000:
        return 9, "giant"
    else:
        return 10, "colossal"
