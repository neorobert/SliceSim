import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass
from typing import List, Dict
import json

# Set page config
st.set_page_config(
    page_title="5G Network Slicing Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class NetworkSlice:
    """Define a network slice with its characteristics"""
    name: str
    bandwidth: float  # Hz
    latency_target: float  # seconds
    reliability_target: float  # percentage
    priority: int  # 1 = highest, 3 = lowest
    allocated_resources: float = 0.0
    current_users: int = 0
    throughput: float = 0.0
    actual_latency: float = 0.0

class BaseStation:
    """Represents a 5G base station"""
    def __init__(self, total_bandwidth: float, max_users: int):
        self.total_bandwidth = total_bandwidth
        self.max_users = max_users
        self.available_bandwidth = total_bandwidth
        self.connected_users = 0

class NetworkSlicingSimulator:
    """Main simulator for 5G network slicing"""
    
    def __init__(self):
        self.slices = []
        self.base_stations = []
        self.simulation_time = 0
        self.performance_history = {
            'time': [],
            'throughput': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'latency': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'users': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'allocated_bandwidth': {'eMBB': [], 'URLLC': [], 'mMTC': []}
        }
    
    def setup_network_slices(self, embb_config, urllc_config, mmtc_config):
        """Initialize network slices with custom configurations"""
        # Enhanced Mobile Broadband (eMBB)
        embb_slice = NetworkSlice(
            name='eMBB',
            bandwidth=embb_config['bandwidth'] * 1e6,  # Convert MHz to Hz
            latency_target=embb_config['latency_target'] / 1000,  # Convert ms to s
            reliability_target=embb_config['reliability_target'],
            priority=embb_config['priority']
        )
        
        # Ultra-Reliable Low-Latency Communications (URLLC)
        urllc_slice = NetworkSlice(
            name='URLLC',
            bandwidth=urllc_config['bandwidth'] * 1e6,
            latency_target=urllc_config['latency_target'] / 1000,
            reliability_target=urllc_config['reliability_target'],
            priority=urllc_config['priority']
        )
        
        # Massive Machine Type Communications (mMTC)
        mmtc_slice = NetworkSlice(
            name='mMTC',
            bandwidth=mmtc_config['bandwidth'] * 1e6,
            latency_target=mmtc_config['latency_target'] / 1000,
            reliability_target=mmtc_config['reliability_target'],
            priority=mmtc_config['priority']
        )
        
        self.slices = [embb_slice, urllc_slice, mmtc_slice]
        return self.slices
    
    def setup_base_stations(self, num_stations: int, total_bandwidth: float):
        """Setup base stations with configurable parameters"""
        self.base_stations = []
        for i in range(num_stations):
            bs = BaseStation(total_bandwidth=total_bandwidth * 1e6, max_users=1000)
            self.base_stations.append(bs)
    
    def generate_traffic(self, slice_name: str, traffic_config: dict) -> int:
        """Generate traffic based on user-defined parameters"""
        base_traffic = traffic_config[slice_name]['base_users']
        variation = traffic_config[slice_name]['variation']

        # Precompute traffic values for efficiency
        poisson_traffic = np.random.poisson(base_traffic, size=1)[0]
        normal_variation = np.random.normal(0, variation, size=1)[0]

        # Generate traffic with Poisson distribution + normal variation
        traffic = max(0, poisson_traffic + normal_variation)
        return int(traffic)

    def dynamic_resource_allocation(self):
        """Implement dynamic resource allocation"""
        total_demand = sum(slice.current_users for slice in self.slices)

        if total_demand == 0:
            return

        total_bandwidth = sum(bs.total_bandwidth for bs in self.base_stations)

        # Precompute factors for efficiency
        base_allocations = np.array([slice_obj.bandwidth / total_bandwidth for slice_obj in self.slices])
        demand_factors = np.array([slice_obj.current_users / total_demand if total_demand > 0 else 0 for slice_obj in self.slices])
        priority_weights = np.array([(4 - slice_obj.priority) / 6 for slice_obj in self.slices])

        for i, slice_obj in enumerate(self.slices):
            slice_obj.allocated_resources = min(
                total_bandwidth * (base_allocations[i] * 0.5 + demand_factors[i] * 0.3 + priority_weights[i] * 0.2),
                total_bandwidth * 0.8
            )
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        for slice_obj in self.slices:
            if slice_obj.current_users > 0:
                spectral_efficiency = self.get_spectral_efficiency(slice_obj.name)
                slice_obj.throughput = slice_obj.allocated_resources * spectral_efficiency
                
                load_factor = slice_obj.current_users / 100
                slice_obj.actual_latency = slice_obj.latency_target * (1 + load_factor * 0.5)
            else:
                slice_obj.throughput = 0
                slice_obj.actual_latency = slice_obj.latency_target
    
    def get_spectral_efficiency(self, slice_name: str) -> float:
        """Get spectral efficiency for different slice types"""
        efficiency_map = {
            'eMBB': 8.0,
            'URLLC': 4.0,
            'mMTC': 2.0
        }
        return efficiency_map.get(slice_name, 5.0)
    
    def run_single_step(self, traffic_config):
        """Run a single simulation step"""
        # Generate traffic
        for slice_obj in self.slices:
            slice_obj.current_users = self.generate_traffic(slice_obj.name, traffic_config)
        
        # Allocate resources
        self.dynamic_resource_allocation()
        
        # Calculate metrics
        self.calculate_performance_metrics()
        
        # Store data
        self.performance_history['time'].append(self.simulation_time)
        for slice_obj in self.slices:
            self.performance_history['throughput'][slice_obj.name].append(slice_obj.throughput / 1e6)
            self.performance_history['latency'][slice_obj.name].append(slice_obj.actual_latency * 1000)
            self.performance_history['users'][slice_obj.name].append(slice_obj.current_users)
            self.performance_history['allocated_bandwidth'][slice_obj.name].append(slice_obj.allocated_resources / 1e6)
        
        self.simulation_time += 1
    
    def reset_simulation(self):
        """Reset simulation data"""
        self.simulation_time = 0
        self.performance_history = {
            'time': [],
            'throughput': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'latency': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'users': {'eMBB': [], 'URLLC': [], 'mMTC': []},
            'allocated_bandwidth': {'eMBB': [], 'URLLC': [], 'mMTC': []}
        }

def main():
    # Title and description
    st.title("📡 5G Network Slicing Simulator")
    st.markdown("**Interactive simulation of 5G network slicing with configurable parameters**")
    st.markdown("**By Neo Robert**")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = NetworkSlicingSimulator()
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0
    
    # Sidebar configuration
    st.sidebar.header("🔧 Network Configuration")
    
    # Base Station Configuration
    st.sidebar.subheader("Base Station Settings")
    num_base_stations = st.sidebar.slider("Number of Base Stations", 1, 10, 3)
    total_bandwidth_mhz = st.sidebar.slider("Total Bandwidth per BS (MHz)", 50, 200, 100)
    
    # Network Slice Configuration
    st.sidebar.subheader("Network Slice Configuration")
    
    # eMBB Configuration
    st.sidebar.markdown("**eMBB Slice**")
    embb_config = {
        'bandwidth': st.sidebar.slider("eMBB Bandwidth (MHz)", 5, 50, 20, key='embb_bw'),
        'latency_target': st.sidebar.slider("eMBB Latency Target (ms)", 5.0, 50.0, 10.0, key='embb_lat'),
        'reliability_target': st.sidebar.slider("eMBB Reliability (%)", 90.0, 99.0, 95.0, key='embb_rel'),
        'priority': st.sidebar.selectbox("eMBB Priority", [1, 2, 3], index=1, key='embb_pri')
    }
    
    # URLLC Configuration
    st.sidebar.markdown("**URLLC Slice**")
    urllc_config = {
        'bandwidth': st.sidebar.slider("URLLC Bandwidth (MHz)", 5, 30, 10, key='urllc_bw'),
        'latency_target': st.sidebar.slider("URLLC Latency Target (ms)", 0.5, 5.0, 1.0, step=0.5, key='urllc_lat'),
        'reliability_target': st.sidebar.slider("URLLC Reliability (%)", 95.0, 99.999, 99.999, key='urllc_rel'),
        'priority': st.sidebar.selectbox("URLLC Priority", [1, 2, 3], index=0, key='urllc_pri')
    }
    
    # mMTC Configuration
    st.sidebar.markdown("**mMTC Slice**")
    mmtc_config = {
        'bandwidth': st.sidebar.slider("mMTC Bandwidth (MHz)", 2, 20, 5, key='mmtc_bw'),
        'latency_target': st.sidebar.slider("mMTC Latency Target (ms)", 50.0, 500.0, 100.0, key='mmtc_lat'),
        'reliability_target': st.sidebar.slider("mMTC Reliability (%)", 85.0, 95.0, 90.0, key='mmtc_rel'),
        'priority': st.sidebar.selectbox("mMTC Priority", [1, 2, 3], index=2, key='mmtc_pri')
    }
    
    # Traffic Configuration
    st.sidebar.subheader("Traffic Configuration")
    traffic_config = {
        'eMBB': {
            'base_users': st.sidebar.slider("eMBB Base Users", 20, 200, 100, key='embb_users'),
            'variation': st.sidebar.slider("eMBB Traffic Variation", 5, 50, 20, key='embb_var')
        },
        'URLLC': {
            'base_users': st.sidebar.slider("URLLC Base Users", 10, 100, 50, key='urllc_users'),
            'variation': st.sidebar.slider("URLLC Traffic Variation", 2, 20, 10, key='urllc_var')
        },
        'mMTC': {
            'base_users': st.sidebar.slider("mMTC Base Users", 50, 500, 200, key='mmtc_users'),
            'variation': st.sidebar.slider("mMTC Traffic Variation", 10, 100, 50, key='mmtc_var')
        }
    }
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Initialize Network", type="primary"):
            st.session_state.simulator.reset_simulation()
            st.session_state.simulator.setup_network_slices(embb_config, urllc_config, mmtc_config)
            st.session_state.simulator.setup_base_stations(num_base_stations, total_bandwidth_mhz)
            st.session_state.step_count = 0
            st.success("Network initialized successfully!")
    
    with col2:
        if st.button("▶️ Run Step"):
            if st.session_state.simulator.slices:
                st.session_state.simulator.run_single_step(traffic_config)
                st.session_state.step_count += 1
                st.info(f"Step {st.session_state.step_count} completed")
            else:
                st.error("Please initialize network first!")
    
    with col3:
        auto_run = st.button("🔄 Auto Run")
        if auto_run:
            st.session_state.simulation_running = not st.session_state.simulation_running
    
    with col4:
        if st.button("🔄 Reset"):
            st.session_state.simulator.reset_simulation()
            st.session_state.step_count = 0
            st.session_state.simulation_running = False
            st.success("Simulation reset!")
    
    # Auto-run functionality
    if st.session_state.simulation_running and st.session_state.simulator.slices:
        time.sleep(0.5)  # Small delay for visualization
        st.session_state.simulator.run_single_step(traffic_config)
        st.session_state.step_count += 1
        st.rerun()
    
    # Display current network status
    if st.session_state.simulator.slices:
        st.header("📊 Current Network Status")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        for i, slice_obj in enumerate(st.session_state.simulator.slices):
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"{slice_obj.name} Users",
                    value=slice_obj.current_users,
                    delta=None
                )
                st.metric(
                    label=f"{slice_obj.name} Throughput",
                    value=f"{slice_obj.throughput/1e6:.1f} Mbps",
                    delta=None
                )
                st.metric(
                    label=f"{slice_obj.name} Latency",
                    value=f"{slice_obj.actual_latency*1000:.2f} ms",
                    delta=f"Target: {slice_obj.latency_target*1000:.1f} ms"
                )
    
    # Performance visualization
    if st.session_state.simulator.performance_history['time']:
        st.header("📈 Performance Analytics")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Throughput", "Latency", "User Load", "Resource Allocation"])
        
        with tab1:
            fig = go.Figure()
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                fig.add_trace(go.Scatter(
                    x=st.session_state.simulator.performance_history['time'],
                    y=st.session_state.simulator.performance_history['throughput'][slice_name],
                    mode='lines+markers',
                    name=slice_name,
                    line=dict(width=3)
                ))
            fig.update_layout(
                title="Throughput Performance Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Throughput (Mbps)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                fig.add_trace(go.Scatter(
                    x=st.session_state.simulator.performance_history['time'],
                    y=st.session_state.simulator.performance_history['latency'][slice_name],
                    mode='lines+markers',
                    name=slice_name,
                    line=dict(width=3)
                ))
            fig.update_layout(
                title="Latency Performance Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Latency (ms)",
                yaxis_type="log",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                fig.add_trace(go.Scatter(
                    x=st.session_state.simulator.performance_history['time'],
                    y=st.session_state.simulator.performance_history['users'][slice_name],
                    mode='lines+markers',
                    name=slice_name,
                    line=dict(width=3)
                ))
            fig.update_layout(
                title="User Load Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Number of Users",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            fig = go.Figure()
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                fig.add_trace(go.Scatter(
                    x=st.session_state.simulator.performance_history['time'],
                    y=st.session_state.simulator.performance_history['allocated_bandwidth'][slice_name],
                    mode='lines+markers',
                    name=slice_name,
                    line=dict(width=3)
                ))
            fig.update_layout(
                title="Allocated Bandwidth Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Allocated Bandwidth (MHz)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        if len(st.session_state.simulator.performance_history['time']) > 1:
            st.subheader("📋 Performance Summary")
            
            summary_data = []
            for slice_obj in st.session_state.simulator.slices:
                slice_name = slice_obj.name
                avg_throughput = np.mean(st.session_state.simulator.performance_history['throughput'][slice_name])
                avg_latency = np.mean(st.session_state.simulator.performance_history['latency'][slice_name])
                avg_users = np.mean(st.session_state.simulator.performance_history['users'][slice_name])
                avg_bandwidth = np.mean(st.session_state.simulator.performance_history['allocated_bandwidth'][slice_name])
                
                sla_met = "✅" if avg_latency <= slice_obj.latency_target * 1000 else "❌"
                
                summary_data.append({
                    'Slice': slice_name,
                    'Avg Throughput (Mbps)': f"{avg_throughput:.2f}",
                    'Avg Latency (ms)': f"{avg_latency:.2f}",
                    'Target Latency (ms)': f"{slice_obj.latency_target * 1000:.1f}",
                    'SLA Met': sla_met,
                    'Avg Users': f"{avg_users:.0f}",
                    'Avg Allocated BW (MHz)': f"{avg_bandwidth:.1f}",
                    'Priority': slice_obj.priority
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**5G Network Slicing Simulator** - Built with Streamlit | Step: " + str(st.session_state.step_count))

if __name__ == "__main__":
    main()