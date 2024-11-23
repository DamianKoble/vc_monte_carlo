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
        
    def run_discrete_simulation(
        self,
        num_simulations: int,
        total_capital: float,
        num_companies: int,
        min_investment: float,
        max_investment: float,
        seed_params: Dict[str, float],
        series_a_params: Dict[str, float],
        existing_investments: Optional[List[ExistingInvestment]] = None
    ):
        results = []
        for _ in range(num_simulations):
            portfolio_value = 0
            remaining_capital = total_capital
            
            # Include existing investments if provided
            if existing_investments:
                for investment in existing_investments:
                    outcome = self._generate_discrete_outcome(
                        investment.stage,
                        seed_params if investment.stage == "Seed" else series_a_params
                    )
                    portfolio_value += investment.amount * outcome
                    remaining_capital -= investment.amount
            
            # Generate new investments
            remaining_companies = num_companies - (len(existing_investments) if existing_investments else 0)
            if remaining_companies > 0:
                investment_size = remaining_capital / remaining_companies
                investment_size = min(max(investment_size, min_investment), max_investment)
                
                for _ in range(remaining_companies):
                    # Randomly choose stage (50-50 split between Seed and Series A)
                    stage = np.random.choice(["Seed", "Series A"])
                    params = seed_params if stage == "Seed" else series_a_params
                    
                    outcome = self._generate_discrete_outcome(stage, params)
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
                
                for _ in range(remaining_companies):
                    stage = np.random.choice(["Seed", "Series A"])
                    alpha = seed_alpha if stage == "Seed" else series_a_alpha
                    max_return = seed_max_return if stage == "Seed" else series_a_max_return
                    
                    outcome = self._generate_power_law_outcome(stage, alpha, max_return)
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
    
    # Initialize simulator
    simulator = VCSimulator()
    
    # Sidebar for common parameters
    st.sidebar.header("Portfolio Parameters")
    total_capital = st.sidebar.number_input("Total Capital (€)", min_value=1000000, value=20000000, step=1000000)
    min_investment = st.sidebar.number_input("Minimum Investment (€)", min_value=100000, value=250000, step=50000)
    max_investment = st.sidebar.number_input("Maximum Investment (€)", min_value=500000, value=3000000, step=100000)
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, value=1000, step=100)
    
    # Existing investments section
    st.sidebar.header("Existing Investments")
    num_existing = st.sidebar.number_input("Number of Existing Investments", min_value=0, value=0, step=1)
    existing_investments = []
    
    if num_existing > 0:
        st.sidebar.subheader("Enter Existing Investments")
        for i in range(num_existing):
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                name = st.text_input(f"Name #{i+1}", value=f"Company {i+1}")
            with col2:
                amount = st.number_input(f"Amount #{i+1} (€)", min_value=0, value=500000)
            with col3:
                stage = st.selectbox(f"Stage #{i+1}", ["Seed", "Series A"], key=f"stage_{i}")
            
            existing_investments.append(
                ExistingInvestment(name=name, amount=amount, stage=stage, date="2024")
            )
    
    # Create tabs
    tab1, tab2 = st.tabs(["Discrete Outcome Model", "Power Law Model"])
    
    with tab1:
        st.header("Discrete Outcome Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Seed Stage Parameters")
            seed_params = {
                "loss_rate": st.slider("Loss Rate (Seed)", 0.0, 1.0, 0.4, 0.05),
                "sideways_rate": st.slider("Sideways Rate (Seed)", 0.0, 1.0, 0.3, 0.05),
                "small_win_rate": st.slider("Small Win Rate (Seed)", 0.0, 1.0, 0.2, 0.05),
                "medium_win_rate": st.slider("Medium Win Rate (Seed)", 0.0, 1.0, 0.08, 0.02),
                "large_win_rate": st.slider("Large Win Rate (Seed)", 0.0, 1.0, 0.02, 0.01),
                "small_win_multiple": st.number_input("Small Win Multiple (Seed)", min_value=1.0, value=3.0),
                "medium_win_multiple": st.number_input("Medium Win Multiple (Seed)", min_value=1.0, value=10.0),
                "large_win_multiple": st.number_input("Large Win Multiple (Seed)", min_value=1.0, value=30.0)
            }
            
        with col2:
            st.subheader("Series A Parameters")
            series_a_params = {
                "loss_rate": st.slider("Loss Rate (Series A)", 0.0, 1.0, 0.3, 0.05),
                "sideways_rate": st.slider("Sideways Rate (Series A)", 0.0, 1.0, 0.35, 0.05),
                "small_win_rate": st.slider("Small Win Rate (Series A)", 0.0, 1.0, 0.25, 0.05),
                "medium_win_rate": st.slider("Medium Win Rate (Series A)", 0.0, 1.0, 0.08, 0.02),
                "large_win_rate": st.slider("Large Win Rate (Series A)", 0.0, 1.0, 0.02, 0.01),
                "small_win_multiple": st.number_input("Small Win Multiple (Series A)", min_value=1.0, value=2.5),
                "medium_win_multiple": st.number_input("Medium Win Multiple (Series A)", min_value=1.0, value=8.0),
                "large_win_multiple": st.number_input("Large Win Multiple (Series A)", min_value=1.0, value=20.0)
            }
        
        num_companies = st.slider("Number of Companies", 
                                min_value=max(len(existing_investments), 1), 
                                max_value=50, 
                                value=10)
        
        if st.button("Run Discrete Simulation"):
            results = simulator.run_discrete_simulation(
                num_simulations=num_simulations,
                total_capital=total_capital,
                num_companies=num_companies,
                min_investment=min_investment,
                max_investment=max_investment,
                seed_params=seed_params,
                series_a_params=series_a_params,
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Seed Stage Parameters")
            seed_alpha = st.slider("Seed Alpha", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
            seed_max_return = st.number_input("Seed Max Return Multiple", min_value=1.0, value=100.0)
            
        with col2:
            st.subheader("Series A Parameters")
            series_a_alpha = st.slider("Series A Alpha", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
            series_a_max_return = st.number_input("Series A Max Return Multiple", min_value=1.0, value=50.0)
        
        num_companies_power = st.slider("Number of Companies (Power Law)", 
                                      min_value=max(len(existing_investments), 1), 
                                      max_value=50, 
                                      value=10)
        
        if st.button("Run Power Law Simulation"):
            results = simulator.run_power_law_simulation(
                num_simulations=num_simulations,
                total_capital=total_capital,
                num_companies=num_companies_power,
                min_investment=min_investment,
                max_investment=max_investment,
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

if __name__ == "__main__":
    main()