import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ExistingInvestment:
    name: str
    amount: float
    stage: str
    date: str

class VCSimulator:
    def __init__(self):
        self.existing_investments: List[ExistingInvestment] = []
    
    # def run_discrete_simulation(
    #     self,
    #     num_simulations: int,
    #     total_capital: float,
    #     num_companies: int,
    #     min_investment: float,
    #     max_investment: float,
    #     preseed_params: Dict[str, float],
    #     seed_params: Dict[str, float],
    #     series_a_params: Dict[str, float],
    #     series_b_params: Dict[str, float],
    #     existing_investments: Optional[List[ExistingInvestment]] = None
    # ):
    #     results = []
    #     for _ in range(num_simulations):
    #         portfolio_value = 0
    #         remaining_capital = total_capital
            
    #         # Include existing investments if provided
    #         if existing_investments:
    #             for investment in existing_investments:
    #                 outcome = self._generate_discrete_outcome(
    #                     investment.stage,
    #                     preseed_params if investment.stage == "Pre-seed"
    #                     else seed_params if investment.stage == "Seed"
    #                     else series_a_params if investment.stage == "Series A"
    #                     else series_b_params
    #                 )
    #                 portfolio_value += investment.amount * outcome
    #                 remaining_capital -= investment.amount
            
    #         # Generate new investments
    #         remaining_companies = num_companies - (len(existing_investments) if existing_investments else 0)
    #         if remaining_companies > 0:
    #             investment_size = remaining_capital / remaining_companies
    #             investment_size = min(max(investment_size, min_investment), max_investment)
                
    #             for _ in range(remaining_companies):
    #                  # Use allocation percentages to determine stage
    #                 rand = np.random.random() * 100
    #                 if rand < preseed_allocation:
    #                     stage = "Pre-seed"
    #                     params = preseed_params
    #                 elif rand < preseed_allocation + seed_allocation:
    #                     stage = "Seed"
    #                     params = seed_params
    #                 elif rand < preseed_allocation + seed_allocation + series_a_allocation:
    #                     stage = "Series A"
    #                     params = series_a_params
    #                 else:
    #                     stage = "Series B"
    #                     params = series_b_params
                    
    #                 outcome = self._generate_discrete_outcome(stage, params)
    #                 portfolio_value += investment_size * outcome
            
    #         moic = portfolio_value / total_capital
    #         results.append(moic)
            
    #     return np.array(results)

    def run_discrete_simulation(
        self,
        num_simulations: int,
        total_capital: float,
        num_companies: int,
        min_investment: float,
        max_investment: float,
        allocations: Dict[str, float],
        preseed_params: Dict[str, float],  # Add this
        seed_params: Dict[str, float],
        series_a_params: Dict[str, float],
        series_b_params: Dict[str, float],  # Add this
        existing_investments: Optional[List[ExistingInvestment]] = None
    ):
        results = []
        stage_params = {
            "Pre-seed": preseed_params,
            "Seed": seed_params,
            "Series A": series_a_params,
            "Series B": series_b_params
        }

        for _ in range(num_simulations):
            portfolio_value = 0
            remaining_capital = total_capital
            
            # Include existing investments if provided
            if existing_investments:
                for investment in existing_investments:
                    outcome = self._generate_discrete_outcome(
                        investment.stage,
                        stage_params[investment.stage]
                    )
                    portfolio_value += investment.amount * outcome
                    remaining_capital -= investment.amount
            
            # Generate new investments
            remaining_companies = num_companies - (len(existing_investments) if existing_investments else 0)
            if remaining_companies > 0:
                investment_size = remaining_capital / remaining_companies
                investment_size = min(max(investment_size, min_investment), max_investment)
                
                # Calculate number of companies per stage
                preseed_companies = int(remaining_companies * allocations['preseed'] / 100)
                seed_companies = int(remaining_companies * allocations['seed'] / 100)
                series_a_companies = int(remaining_companies * allocations['series_a'] / 100)
                series_b_companies = int(remaining_companies * allocations['series_b'] / 100)
                
                # Pre-seed investments
                for _ in range(preseed_companies):
                    outcome = self._generate_discrete_outcome("Pre-seed", preseed_params)
                    portfolio_value += investment_size * outcome
                
                # Seed investments
                for _ in range(seed_companies):
                    outcome = self._generate_discrete_outcome("Seed", seed_params)
                    portfolio_value += investment_size * outcome
                
                # Series A investments
                for _ in range(series_a_companies):
                    outcome = self._generate_discrete_outcome("Series A", series_a_params)
                    portfolio_value += investment_size * outcome
                
                # Series B investments
                for _ in range(series_b_companies):
                    outcome = self._generate_discrete_outcome("Series B", series_b_params)
                    portfolio_value += investment_size * outcome
            
            moic = portfolio_value / total_capital
            results.append(moic)
            
        return np.array(results)

    def _generate_discrete_outcome(self, stage: str, params: Dict[str, float]) -> float:
        """Generate outcome based on discrete probability distribution"""
        outcomes = {
            0.0: params["loss_rate"],
            1.0: params["sideways_rate"],
            params["small_win_multiple"]: params["small_win_rate"],
            params["medium_win_multiple"]: params["medium_win_rate"],
            params["large_win_multiple"]: params["large_win_rate"]
        }
        return np.random.choice(
            list(outcomes.keys()),
            p=list(outcomes.values())
        )    

    def run_power_law_simulation(
        self,
        num_simulations: int,
        total_capital: float,
        num_companies: int,
        min_investment: float,
        max_investment: float,
        allocations: Dict[str, float],
        seed_alpha: float,
        series_a_alpha: float,
        seed_max_return: float,
        series_a_max_return: float,
        existing_investments: Optional[List[ExistingInvestment]] = None
    ):
        results = []
        for _ in range(num_simulations):
            portfolio_value = 0
            remaining_capital = total_capital
            
            # Include existing investments if provided
            if existing_investments:
                for investment in existing_investments:
                    outcome = self._generate_power_law_outcome(
                        investment.stage,
                        seed_alpha if investment.stage == "Seed" else series_a_alpha,
                        seed_max_return if investment.stage == "Seed" else series_a_max_return
                    )
                    portfolio_value += investment.amount * outcome
                    remaining_capital -= investment.amount
            
            # Generate new investments
            remaining_companies = num_companies - (len(existing_investments) if existing_investments else 0)
            if remaining_companies > 0:
                investment_size = remaining_capital / remaining_companies
                investment_size = min(max(investment_size, min_investment), max_investment)
                
                # Calculate number of companies per stage based on allocations
                preseed_companies = int(remaining_companies * allocations['preseed'] / 100)
                seed_companies = int(remaining_companies * allocations['seed'] / 100)
                series_a_companies = int(remaining_companies * allocations['series_a'] / 100)
                series_b_companies = int(remaining_companies * allocations['series_b'] / 100)
                
                # Simulate each stage
                for _ in range(preseed_companies):
                    outcome = self._generate_power_law_outcome("Pre-seed", seed_alpha, seed_max_return * 1.5)
                    portfolio_value += investment_size * outcome
                    
                for _ in range(seed_companies):
                    outcome = self._generate_power_law_outcome("Seed", seed_alpha, seed_max_return)
                    portfolio_value += investment_size * outcome
                    
                for _ in range(series_a_companies):
                    outcome = self._generate_power_law_outcome("Series A", series_a_alpha, series_a_max_return)
                    portfolio_value += investment_size * outcome
                    
                for _ in range(series_b_companies):
                    outcome = self._generate_power_law_outcome("Series B", series_a_alpha, series_a_max_return * 0.5)
                    portfolio_value += investment_size * outcome
            
            moic = portfolio_value / total_capital
            results.append(moic)
            
        return np.array(results)
    
    def _generate_power_law_outcome(self, stage: str, alpha: float, max_return: float) -> float:
        """Generate outcome based on power law distribution"""
        x = np.random.uniform(0, 1)
        # Add small constant to avoid division by zero
        return min(max_return, (1 / (1 - x + 1e-10)) ** (1 / alpha))

def main():
    st.set_page_config(layout="wide")
    st.title("VC Portfolio Monte Carlo Simulator")
    
    # Sidebar for common parameters
    st.sidebar.header("Portfolio Parameters")
    num_companies = st.sidebar.number_input(
        "Total Number of Companies",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )
    total_capital = st.sidebar.number_input("Total Capital (€)", min_value=1, value=20000000, step=1000000)
    min_investment = st.sidebar.number_input("Minimum Investment (€)", min_value=1, value=250000, step=50000)
    max_investment = st.sidebar.number_input("Maximum Investment (€)", min_value=1, value=3000000, step=100000)
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, value=1000, step=100)
    
    # Existing investments section
    st.sidebar.header("Existing Investments")
    num_existing = st.sidebar.number_input("Number of Existing Investments", min_value=0, value=0, step=1)
    existing_investments = []

    # Initialize simulator
    simulator = VCSimulator()

    
    col_alloc1, col_alloc2 = st.columns(2)
    
    with col_alloc1:
        st.subheader("Stage Allocation")
        preseed_allocation = st.number_input(
            "Pre-seed Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=20,
            step=5,
            key="preseed_alloc"
        )
        
        seed_allocation = st.number_input(
            "Seed Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=30,
            step=5,
            key="seed_alloc"
        )
        
        series_a_allocation = st.number_input(
            "Series A Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=30,
            step=5,
            key="series_a_alloc"
        )
        
        series_b_allocation = st.number_input(
            "Series B Allocation (%)", 
            min_value=0, 
            max_value=100, 
            value=20,
            step=5,
            key="series_b_alloc"
        )
        
        total_allocation = preseed_allocation + seed_allocation + series_a_allocation + series_b_allocation
    
    with col_alloc2:
            st.markdown("### Allocation Summary")
            # Color code the total based on whether it equals 100
            if total_allocation == 100:
                total_color = "green"
            else:
                total_color = "red"
                
            st.markdown(f"""
            Current allocation:
            - Pre-seed: {preseed_allocation}%
            - Seed: {seed_allocation}%
            - Series A: {series_a_allocation}%
            - Series B: {series_b_allocation}%
            
            **Total: <span style='color: {total_color}'>{total_allocation}%</span>**
            """, unsafe_allow_html=True)
            
            if total_allocation != 100:
                st.error("Total allocation must equal 100%")
            else:
                st.markdown("### Company Distribution")
                st.markdown(f"""
                - Pre-seed: {int(num_companies * preseed_allocation/100)} companies
                - Seed: {int(num_companies * seed_allocation/100)} companies
                - Series A: {int(num_companies * series_a_allocation/100)} companies
                - Series B: {int(num_companies * series_b_allocation/100)} companies
                """)    
        
    if num_existing > 0:
        st.sidebar.subheader("Enter Existing Investments")
        for i in range(num_existing):
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                name = st.text_input(f"Name #{i+1}", value=f"Company {i+1}")
            with col2:
                amount = st.number_input(f"Amount #{i+1} (€)", min_value=0, value=500000)
            with col3:
                stage = st.selectbox(f"Stage #{i+1}", ["Pre-seed", "Seed", "Series A", "Series B"], key=f"stage_{i}")
            
            existing_investments.append(
                ExistingInvestment(name=name, amount=amount, stage=stage, date="2024")
            )
    
    # Create tabs
    if total_allocation == 100:
        tab1, tab2 = st.tabs(["Discrete Outcome Model", "Power Law Model"])
    
        with tab1:
            st.header("Discrete Outcome Model")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.subheader("Pre-seed Parameters")
                preseed_params = {
                    "loss_rate": st.slider("Loss Rate (Pre-seed)", 0.0, 1.0, 0.5, 0.05),
                    "sideways_rate": st.slider("Sideways Rate (Pre-seed)", 0.0, 1.0, 0.25, 0.05),
                    "small_win_rate": st.slider("Small Win Rate (Pre-seed)", 0.0, 1.0, 0.15, 0.05),
                    "medium_win_rate": st.slider("Medium Win Rate (Pre-seed)", 0.0, 1.0, 0.08, 0.02),
                    "large_win_rate": st.slider("Large Win Rate (Pre-seed)", 0.0, 1.0, 0.02, 0.01),
                    "small_win_multiple": st.number_input("Small Win Multiple (Pre-seed)", min_value=1.0, value=4.0),
                    "medium_win_multiple": st.number_input("Medium Win Multiple (Pre-seed)", min_value=1.0, value=15.0),
                    "large_win_multiple": st.number_input("Large Win Multiple (Pre-seed)", min_value=1.0, value=70.0)
                }

            with col2:
                st.subheader("Seed Stage Parameters")
                seed_params = {
                    "loss_rate": st.slider("Loss Rate (Seed)", 0.0, 1.0, 0.4, 0.05),
                    "sideways_rate": st.slider("Sideways Rate (Seed)", 0.0, 1.0, 0.3, 0.05),
                    "small_win_rate": st.slider("Small Win Rate (Seed)", 0.0, 1.0, 0.2, 0.05),
                    "medium_win_rate": st.slider("Medium Win Rate (Seed)", 0.0, 1.0, 0.08, 0.02),
                    "large_win_rate": st.slider("Large Win Rate (Seed)", 0.0, 1.0, 0.02, 0.01),
                    "small_win_multiple": st.number_input("Small Win Multiple (Seed)", min_value=1.0, value=3.0),
                    "medium_win_multiple": st.number_input("Medium Win Multiple (Seed)", min_value=1.0, value=10.0),
                    "large_win_multiple": st.number_input("Large Win Multiple (Seed)", min_value=1.0, value=50.0)
                }
                
            with col3:
                st.subheader("Series A Parameters")
                series_a_params = {
                    "loss_rate": st.slider("Loss Rate (Series A)", 0.0, 1.0, 0.3, 0.05),
                    "sideways_rate": st.slider("Sideways Rate (Series A)", 0.0, 1.0, 0.35, 0.05),
                    "small_win_rate": st.slider("Small Win Rate (Series A)", 0.0, 1.0, 0.25, 0.05),
                    "medium_win_rate": st.slider("Medium Win Rate (Series A)", 0.0, 1.0, 0.08, 0.02),
                    "large_win_rate": st.slider("Large Win Rate (Series A)", 0.0, 1.0, 0.02, 0.01),
                    "small_win_multiple": st.number_input("Small Win Multiple (Series A)", min_value=1.0, value=2.5),
                    "medium_win_multiple": st.number_input("Medium Win Multiple (Series A)", min_value=1.0, value=8.0),
                    "large_win_multiple": st.number_input("Large Win Multiple (Series A)", min_value=1.0, value=30.0)
                }

            with col4:
                st.subheader("Series B Parameters")
                series_b_params = {
                    "loss_rate": st.slider("Loss Rate (Series B)", 0.0, 1.0, 0.2, 0.05),
                    "sideways_rate": st.slider("Sideways Rate (Series B)", 0.0, 1.0, 0.4, 0.05),
                    "small_win_rate": st.slider("Small Win Rate (Series B)", 0.0, 1.0, 0.3, 0.05),
                    "medium_win_rate": st.slider("Medium Win Rate (Series B)", 0.0, 1.0, 0.08, 0.02),
                    "large_win_rate": st.slider("Large Win Rate (Series B)", 0.0, 1.0, 0.02, 0.01),
                    "small_win_multiple": st.number_input("Small Win Multiple (Series B)", min_value=1.0, value=2.0),
                    "medium_win_multiple": st.number_input("Medium Win Multiple (Series B)", min_value=1.0, value=5.0),
                    "large_win_multiple": st.number_input("Large Win Multiple (Series B)", min_value=1.0, value=10.0)
                }    
            
            # num_companies = st.slider("Number of Companies", 
            #                         min_value=max(len(existing_investments), 1), 
            #                         max_value=50, 
            #                         value=10)
            
            if st.button("Run Discrete Simulation", key="run_discrete_sim"):
                if total_allocation != 100:
                    st.error("Please adjust allocations to total 100% before running simulation")
                else:
                    allocations = {
                        'preseed': preseed_allocation,
                        'seed': seed_allocation,
                        'series_a': series_a_allocation,
                        'series_b': series_b_allocation
                    }
                    results = simulator.run_discrete_simulation(
                        num_simulations=num_simulations,
                        total_capital=total_capital,
                        num_companies=num_companies,
                        min_investment=min_investment,
                        max_investment=max_investment,
                        allocations=allocations,
                        preseed_params=preseed_params,  # Add this
                        seed_params=seed_params,
                        series_a_params=series_a_params,
                        series_b_params=series_b_params,  # Add this
                        existing_investments=existing_investments if existing_investments else None
                    )
                
                mean_moic = np.mean(results)
                median_moic = np.median(results)
                percentile_95 = np.percentile(results, 95)
                percentile_5 = np.percentile(results, 5)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean MOIC", f"{mean_moic:.2f}x")
                col2.metric("Median MOIC", f"{median_moic:.2f}x")
                col3.metric("95th Percentile", f"{percentile_95:.2f}x")
                col4.metric("5th Percentile", f"{percentile_5:.2f}x")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=results, nbinsx=50, name="MOIC Distribution"))
                fig.add_vline(x=4.0, line_dash="dash", line_color="red", annotation_text="Target 4.0x")
                fig.update_layout(
                    title="Distribution of Portfolio MOIC",
                    xaxis_title="MOIC",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                success_rate = np.mean(results >= 4.0) * 100
                st.metric("Probability of Achieving 4.0x MOIC", f"{success_rate:.1f}%")

            # Export simulation results
                results_df = pd.DataFrame({
                    'simulation': range(1, len(results) + 1),
                    'moic': results,
                    'achieved_target': results >= 4.0,
                    'portfolio_value': results * total_capital,
                    'num_companies': num_companies,
                    'total_capital': total_capital
                })
                
                # Add existing portfolio details if any
                if existing_investments:
                    results_df['num_existing_investments'] = len(existing_investments)
                    results_df['existing_portfolio_value'] = sum(inv.amount for inv in existing_investments)
                else:
                    results_df['num_existing_investments'] = 0
                    results_df['existing_portfolio_value'] = 0
                
                # Add model-specific parameters
                if tab1:  # Discrete Model
                    results_df['model_type'] = 'Discrete'
                    results_df['seed_loss_rate'] = seed_params['loss_rate']
                    results_df['seed_large_win_rate'] = seed_params['large_win_rate']
                    results_df['series_a_loss_rate'] = series_a_params['loss_rate']
                    results_df['series_a_large_win_rate'] = series_a_params['large_win_rate']
                else:  # Power Law Model
                    results_df['model_type'] = 'Power Law'
                    results_df['seed_alpha'] = seed_alpha
                    results_df['series_a_alpha'] = series_a_alpha
                    results_df['seed_max_return'] = seed_max_return
                    results_df['series_a_max_return'] = series_a_max_return
                
                # Create summary statistics
                summary_df = pd.DataFrame([{
                    'mean_moic': mean_moic,
                    'median_moic': median_moic,
                    'percentile_95': percentile_95,
                    'percentile_5': percentile_5,
                    'success_rate': success_rate,
                    'num_simulations': num_simulations,
                    'total_capital': total_capital,
                    'num_companies': num_companies,
                    'num_existing_investments': len(existing_investments) if existing_investments else 0
                }])
                
                # Add download buttons
                st.markdown("### Download Simulation Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_detailed = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Detailed Results",
                        data=csv_detailed,
                        file_name="simulation_detailed_results.csv",
                        mime="text/csv"
                    )
                    st.markdown(f"Detailed results: {len(results_df):,} rows × {len(results_df.columns)} columns")
                
                with col2:
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary Statistics",
                        data=csv_summary,
                        file_name="simulation_summary_results.csv",
                        mime="text/csv"
                    )
                    st.markdown(f"Summary results: {len(summary_df):,} rows × {len(summary_df.columns)} columns")
                
                # Display sample of results
                st.markdown("### Sample of Detailed Results")
                st.dataframe(results_df.head())    

        
        with tab2:
            st.header("Power Law Model")
            
            col1, col2, col3, col4 = st.columns(4)
            
            
            with col1:
                st.subheader("Pre-seed Parameters")
                preseed_alpha = st.slider("Pre-seed Alpha", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
                preseed_max_return = st.number_input("Pre-seed Max Return Multiple", min_value=1.0, value=150.0)

            with col2:
                st.subheader("Seed Stage Parameters")
                seed_alpha = st.slider("Seed Alpha", min_value=0.5, max_value=3.0, value=1.8, step=0.1)
                seed_max_return = st.number_input("Seed Max Return Multiple", min_value=1.0, value=100.0)
                
            with col3:
                st.subheader("Series A Parameters")
                series_a_alpha = st.slider("Series A Alpha", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
                series_a_max_return = st.number_input("Series A Max Return Multiple", min_value=1.0, value=50.0)
            
            with col4:
                st.subheader("Series B Parameters")
                series_b_alpha = st.slider("Series B Alpha", min_value=0.5, max_value=3.0, value=2.5, step=0.1)
                series_b_max_return = st.number_input("Series B Max Return Multiple", min_value=1.0, value=20.0)
            
            if st.button("Run Power Law Simulation", key="run_power_law_sim"):
                if total_allocation != 100:
                    st.error("Please adjust allocations to total 100% before running simulation")
                else:
                    allocations = {
                        'preseed': preseed_allocation,
                        'seed': seed_allocation,
                        'series_a': series_a_allocation,
                        'series_b': series_b_allocation
                    }
                    results = simulator.run_power_law_simulation(
                        num_simulations=num_simulations,
                        total_capital=total_capital,
                        num_companies=num_companies,
                        min_investment=min_investment,
                        max_investment=max_investment,
                        allocations=allocations,  # Add this
                        seed_alpha=seed_alpha,
                        series_a_alpha=series_a_alpha,
                        seed_max_return=seed_max_return,
                        series_a_max_return=series_a_max_return,
                        existing_investments=existing_investments if existing_investments else None
                    )

                mean_moic = np.mean(results)
                median_moic = np.median(results)
                percentile_95 = np.percentile(results, 95)
                percentile_5 = np.percentile(results, 5)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean MOIC", f"{mean_moic:.2f}x")
                col2.metric("Median MOIC", f"{median_moic:.2f}x")
                col3.metric("95th Percentile", f"{percentile_95:.2f}x")
                col4.metric("5th Percentile", f"{percentile_5:.2f}x")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=results, nbinsx=50, name="MOIC Distribution"))
                fig.add_vline(x=4.0, line_dash="dash", line_color="red", annotation_text="Target 4.0x")
                fig.update_layout(
                    title="Distribution of Portfolio MOIC",
                    xaxis_title="MOIC",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                success_rate = np.mean(results >= 4.0) * 100
                st.metric("Probability of Achieving 4.0x MOIC", f"{success_rate:.1f}%")

            # Export simulation results
                results_df = pd.DataFrame({
                    'simulation': range(1, len(results) + 1),
                    'moic': results,
                    'achieved_target': results >= 4.0,
                    'portfolio_value': results * total_capital,
                    'num_companies': num_companies,
                    'total_capital': total_capital
                })
                
                # Add existing portfolio details if any
                if existing_investments:
                    results_df['num_existing_investments'] = len(existing_investments)
                    results_df['existing_portfolio_value'] = sum(inv.amount for inv in existing_investments)
                else:
                    results_df['num_existing_investments'] = 0
                    results_df['existing_portfolio_value'] = 0
                
                # Add model-specific parameters
                if tab1:  # Discrete Model
                    results_df['model_type'] = 'Discrete'
                    results_df['seed_loss_rate'] = seed_params['loss_rate']
                    results_df['seed_large_win_rate'] = seed_params['large_win_rate']
                    results_df['series_a_loss_rate'] = series_a_params['loss_rate']
                    results_df['series_a_large_win_rate'] = series_a_params['large_win_rate']
                else:  # Power Law Model
                    results_df['model_type'] = 'Power Law'
                    results_df['seed_alpha'] = seed_alpha
                    results_df['series_a_alpha'] = series_a_alpha
                    results_df['seed_max_return'] = seed_max_return
                    results_df['series_a_max_return'] = series_a_max_return
                
                # Create summary statistics
                summary_df = pd.DataFrame([{
                    'mean_moic': mean_moic,
                    'median_moic': median_moic,
                    'percentile_95': percentile_95,
                    'percentile_5': percentile_5,
                    'success_rate': success_rate,
                    'num_simulations': num_simulations,
                    'total_capital': total_capital,
                    'num_companies': num_companies,
                    'num_existing_investments': len(existing_investments) if existing_investments else 0
                }])
                
                # Add download buttons
                st.markdown("### Download Simulation Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_detailed = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Detailed Results",
                        data=csv_detailed,
                        file_name="simulation_detailed_results.csv",
                        mime="text/csv"
                    )
                    st.markdown(f"Detailed results: {len(results_df):,} rows × {len(results_df.columns)} columns")
                
                with col2:
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary Statistics",
                        data=csv_summary,
                        file_name="simulation_summary_results.csv",
                        mime="text/csv"
                    )
                    st.markdown(f"Summary results: {len(summary_df):,} rows × {len(summary_df.columns)} columns")
                
                # Display sample of results
                st.markdown("### Sample of Detailed Results")
                st.dataframe(results_df.head())    
    else:
        st.warning("Please adjust stage allocations to total 100% before running simulations")            
if __name__ == "__main__":
    main()