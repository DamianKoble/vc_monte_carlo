import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from itertools import product

@dataclass
class Investment:
    stage: str
    amount: float
    probability_of_success: float
    expected_multiple_range: Tuple[float, float]
    holding_period_range: Tuple[int, int]

@dataclass
class ExistingInvestment:
    stage: str
    amount_invested: float
    current_value: float
    time_held: int
    expected_holding_period: int

class VCPortfolioOptimizer:
    def __init__(self, total_capital: float, target_multiple: float = 4.0):
        self.total_capital = total_capital
        self.target_multiple = target_multiple
        
        # Investment parameters for each stage
        self.seed_params = Investment(
            stage='seed',
            amount=500000,
            probability_of_success=0.20,
            expected_multiple_range=(0, 20),
            holding_period_range=(5, 8)
        )
        
        self.series_a_params = Investment(
            stage='series_a',
            amount=2000000,
            probability_of_success=0.35,
            expected_multiple_range=(0, 10),
            holding_period_range=(4, 7)
        )
        
        self.series_b_params = Investment(
            stage='series_b',
            amount=5000000,
            probability_of_success=0.45,
            expected_multiple_range=(0, 6),
            holding_period_range=(3, 6)
        )
        
        self.existing_portfolio = []

    def add_existing_investment(self, stage: str, amount_invested: float, 
                              current_value: float, time_held: int, 
                              expected_holding_period: int):
        """Add an existing investment to the portfolio"""
        investment = ExistingInvestment(
            stage=stage,
            amount_invested=amount_invested,
            current_value=current_value,
            time_held=time_held,
            expected_holding_period=expected_holding_period
        )
        self.existing_portfolio.append(investment)

    def simulate_investment(self, params: Investment) -> Dict:
        """Simulate a single investment outcome"""
        success = np.random.random() < params.probability_of_success
        
        if success:
            multiple = np.random.uniform(*params.expected_multiple_range)
            holding_period = np.random.randint(*params.holding_period_range)
        else:
            multiple = 0
            holding_period = np.random.randint(*params.holding_period_range)
            
        return {
            'success': success,
            'multiple': multiple,
            'holding_period': holding_period,
            'return': params.amount * multiple
        }

    def simulate_existing_investment(self, investment: ExistingInvestment) -> Dict:
        """Simulate future outcome for an existing investment"""
        # Get parameters based on investment stage
        params = getattr(self, f"{investment.stage}_params")
        
        # Adjust probability of success based on current value vs. initial investment
        current_multiple = investment.current_value / investment.amount_invested
        if current_multiple > 1:
            # Increase success probability for investments performing well
            adjusted_prob = min(params.probability_of_success * 1.5, 0.9)
        else:
            adjusted_prob = params.probability_of_success
        
        success = np.random.random() < adjusted_prob
        
        if success:
            # Adjust multiple range based on current performance
            min_multiple = max(1, current_multiple)
            max_multiple = params.expected_multiple_range[1]
            multiple = np.random.uniform(min_multiple, max_multiple)
            
            # Adjust holding period based on time already held
            remaining_time = investment.expected_holding_period - investment.time_held
            holding_period = investment.time_held + max(0, np.random.randint(0, remaining_time))
        else:
            multiple = min(0.5, current_multiple)  # Allow for some value recovery even in failure
            holding_period = investment.time_held + 1
        
        return {
            'success': success,
            'multiple': multiple,
            'holding_period': holding_period,
            'return': investment.amount_invested * multiple
        }

    def simulate_portfolio(self, seed_allocation: float, series_a_allocation: float, 
                         num_companies: int, num_simulations: int = 1000) -> pd.DataFrame:
        """Simulate portfolio with given parameters"""
        results = []
        
        # Calculate number of companies for each stage
        seed_companies = int(num_companies * seed_allocation)
        series_a_companies = int(num_companies * series_a_allocation)
        series_b_companies = num_companies - seed_companies - series_a_companies
        
        # Calculate total required capital
        total_required_capital = (
            seed_companies * self.seed_params.amount +
            series_a_companies * self.series_a_params.amount +
            series_b_companies * self.series_b_params.amount
        )
        
        if total_required_capital > self.total_capital:
            return pd.DataFrame()
        
        for sim in range(num_simulations):
            portfolio_return = 0
            total_invested = total_required_capital
            successful_exits = 0
            
            # Simulate existing portfolio
            existing_portfolio_return = 0
            existing_portfolio_invested = 0
            for investment in self.existing_portfolio:
                outcome = self.simulate_existing_investment(investment)
                existing_portfolio_return += outcome['return']
                existing_portfolio_invested += investment.amount_invested
                if outcome['success']:
                    successful_exits += 1
            
            # Simulate new investments by stage
            for _ in range(seed_companies):
                outcome = self.simulate_investment(self.seed_params)
                portfolio_return += outcome['return']
                if outcome['success']:
                    successful_exits += 1
            
            for _ in range(series_a_companies):
                outcome = self.simulate_investment(self.series_a_params)
                portfolio_return += outcome['return']
                if outcome['success']:
                    successful_exits += 1
            
            for _ in range(series_b_companies):
                outcome = self.simulate_investment(self.series_b_params)
                portfolio_return += outcome['return']
                if outcome['success']:
                    successful_exits += 1
            
            # Combine existing and new portfolio results
            total_return = portfolio_return + existing_portfolio_return
            total_invested = total_required_capital + existing_portfolio_invested
            total_companies = num_companies + len(self.existing_portfolio)
            
            results.append({
                'simulation': sim + 1,
                'portfolio_return': total_return,
                'total_invested': total_invested,
                'multiple': total_return / total_invested if total_invested > 0 else 0,
                'successful_exits': successful_exits,
                'success_rate': successful_exits / total_companies,
                'seed_allocation': seed_allocation,
                'series_a_allocation': series_a_allocation,
                'series_b_allocation': 1 - seed_allocation - series_a_allocation,
                'num_companies': total_companies,
                'seed_companies': seed_companies,
                'series_a_companies': series_a_companies,
                'series_b_companies': series_b_companies,
                'existing_portfolio_return': existing_portfolio_return,
                'new_portfolio_return': portfolio_return
            })
            
        return pd.DataFrame(results)

    def optimize_portfolio(self) -> Dict:
        """Find optimal portfolio configuration to achieve target multiple"""
        seed_allocations = np.arange(0.2, 0.6, 0.1)
        series_a_allocations = np.arange(0.2, 0.5, 0.1)
        company_counts = range(10, 31, 5)
        
        best_config = None
        best_probability = 0
        all_results = []
        
        for seed_alloc, series_a_alloc, num_companies in product(
            seed_allocations, series_a_allocations, company_counts):
            
            # Skip invalid allocation combinations
            if seed_alloc + series_a_alloc >= 1:
                continue
                
            results = self.simulate_portfolio(seed_alloc, series_a_alloc, num_companies)
            
            if results.empty:
                continue
            
            prob_target = (results['multiple'] >= self.target_multiple).mean()
            mean_multiple = results['multiple'].mean()
            std_multiple = results['multiple'].std()
            
            config_results = {
                'seed_allocation': seed_alloc,
                'series_a_allocation': series_a_alloc,
                'series_b_allocation': 1 - seed_alloc - series_a_alloc,
                'num_companies': num_companies,
                'prob_target': prob_target,
                'mean_multiple': mean_multiple,
                'std_multiple': std_multiple,
                'seed_companies': int(num_companies * seed_alloc),
                'series_a_companies': int(num_companies * series_a_alloc),
                'series_b_companies': int(num_companies * (1 - seed_alloc - series_a_alloc))
            }
            
            all_results.append(config_results)
            
            if prob_target > best_probability:
                best_probability = prob_target
                best_config = config_results
        
        return {
            'best_config': best_config,
            'all_results': pd.DataFrame(all_results)
        }

    def generate_recommendation(self, optimization_results: Dict) -> str:
        """Generate a detailed recommendation based on optimization results"""
        best = optimization_results['best_config']
        
        # Calculate capital requirements
        seed_capital = best['seed_companies'] * self.seed_params.amount
        series_a_capital = best['series_a_companies'] * self.series_a_params.amount
        series_b_capital = best['series_b_companies'] * self.series_b_params.amount
        new_portfolio_capital = seed_capital + series_a_capital + series_b_capital
        
        existing_portfolio_value = sum(inv.current_value for inv in self.existing_portfolio)
        existing_portfolio_cost = sum(inv.amount_invested for inv in self.existing_portfolio)
        
        recommendation = f"""Portfolio Recommendation for {self.target_multiple}x Return Target:

1. Existing Portfolio Summary:
   - Number of companies: {len(self.existing_portfolio)}
   - Total invested: ${existing_portfolio_cost:,.2f}
   - Current value: ${existing_portfolio_value:,.2f}
   - Current multiple: {existing_portfolio_value/existing_portfolio_cost:.2f}x

2. Recommended New Investments:
   - Total New Companies: {best['num_companies']}
   - Seed Stage: {best['seed_companies']} companies ({best['seed_allocation']:.0%})
   - Series A: {best['series_a_companies']} companies ({best['series_a_allocation']:.0%})
   - Series B: {best['series_b_companies']} companies ({best['series_b_allocation']:.0%})

3. Expected Performance (Combined Portfolio):
   - Probability of hitting {self.target_multiple}x target: {best['prob_target']:.1%}
   - Expected multiple: {best['mean_multiple']:.2f}x
   - Standard deviation: {best['std_multiple']:.2f}x

4. Capital Requirements:
   - New capital needed: ${new_portfolio_capital:,.2f}
   - Seed allocation: ${seed_capital:,.2f}
   - Series A allocation: ${series_a_capital:,.2f}
   - Series B allocation: ${series_b_capital:,.2f}

5. Risk Considerations:
   - Portfolio diversification across three stages
   - {best['seed_companies']} seed companies for high upside potential
   - {best['series_a_companies']} Series A companies for growth
   - {best['series_b_companies']} Series B companies for more stable returns
   - Existing portfolio provides base performance
"""
        return recommendation
