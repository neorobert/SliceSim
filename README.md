# SliceSim
Network slicing simulator 


Traffic Generation: The generate_traffic function uses both Poisson and Normal distributions

Dynamic Resource Allocation: The dynamic_resource_allocation function iterates over slices and calculates allocations. 

Performance Metrics Calculation: The calculate_performance_metrics function calculates metrics for each slice.

Streamlit UI Updates: Streamlit's st.rerun() and frequent updates can slow down the UI. Minimize unnecessary reruns and optimize the rendering logic.
Data Storage:

The performance_history dictionary stores data for each time step. If the simulation runs for many steps, this can become a bottleneck. Consider using a more efficient data structure or limiting the history size.
Let me start by optimizing the generate_traffic and dynamic_resource_allocation functions.

