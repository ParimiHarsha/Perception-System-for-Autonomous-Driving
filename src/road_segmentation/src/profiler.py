import cProfile
import pstats
from SAMv2Ros import RoadSegmentation  # Replace with your actual module name
import rospy 
def main():
    # Run your ROS node or the main entry point for your module
        # Initialize the ROS node
    rospy.init_node('road_segmentation_node',anonymous=False)
    road_segmenter = RoadSegmentation()
    # Keep the node running until interrupted
    rospy.spin()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Start your code execution
    main()

    profiler.disable()

    # Print profiling results
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(100)  # Print the top 10 functions by cumulative time
