# app.py - Enhanced Stock Zodiac Analysis Program
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from datetime import datetime, timedelta, date
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ======================
# STOCK ZODIAC ANALYSIS
# ======================
class StockZodiacAnalysis:
    def __init__(self, stock_name, zodiac_sign=None, incorporation_date=None):
        """
        Initialize the stock zodiac analysis
        
        Parameters:
        - stock_name: Name of the stock
        - zodiac_sign: Zodiac sign of the stock (optional, will be determined if not provided)
        - incorporation_date: Date when the stock was incorporated (optional)
        """
        self.stock_name = stock_name
        self.incorporation_date = incorporation_date if incorporation_date else datetime.now().date()
        
        # Determine zodiac sign if not provided
        if zodiac_sign:
            self.zodiac_sign = zodiac_sign
        else:
            self.zodiac_sign = self._determine_zodiac_sign(stock_name)
        
        # Get zodiac sign angle
        self.zodiac_angle = self._get_zodiac_angle(self.zodiac_sign)
        
        # Get current planetary positions
        self.planetary_positions = self._get_planetary_positions()
        
        # Get current aspects to the stock's zodiac sign
        self.aspects = self._check_aspects_to_sign(self.zodiac_angle)
        
        # Initialize analysis results
        self.trend_prediction = None
        self.best_month = None
        self.critical_dates = None
        self.monthly_analysis = None
        self.price_projections = None
    
    def _determine_zodiac_sign(self, stock_name):
        """
        Determine the zodiac sign based on the stock name
        This is a simplified approach - in a real application, you might use more sophisticated methods
        """
        # Convert stock name to uppercase for consistent processing
        stock_name = stock_name.upper()
        
        # Simple hash function to map stock name to zodiac sign
        # This is just an example - you can use any method you prefer
        hash_value = sum(ord(char) for char in stock_name)
        zodiac_index = hash_value % 12
        
        zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                       'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        
        return zodiac_signs[zodiac_index]
    
    def _get_zodiac_from_angle(self, angle):
        """Convert angle to zodiac sign"""
        zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                   'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        index = int(angle // 30) % 12
        return zodiacs[index]
    
    def _get_zodiac_angle(self, zodiac_sign):
        """Convert zodiac sign to angle"""
        zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                   'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        if zodiac_sign in zodiacs:
            return zodiacs.index(zodiac_sign) * 30
        return 0  # Default to Aries if not found
    
    def _get_planetary_positions(self):
        """Get current planetary positions"""
        # In a real implementation, you would use ephem or another library
        # For this example, we'll use pre-calculated positions
        return {
            'Sun': 132.5,    # Leo
            'Moon': 45.2,    # Taurus
            'Mercury': 155.8, # Virgo
            'Venus': 210.3,  # Libra
            'Mars': 85.7,    # Gemini
            'Jupiter': 55.4,  # Taurus
            'Saturn': 355.2,  # Pisces
            'Uranus': 27.8,   # Aries
            'Neptune': 352.1, # Pisces
            'Pluto': 298.5    # Capricorn
        }
    
    def _calculate_aspect(self, angle1, angle2):
        """Calculate the angular distance between two points"""
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _check_aspects_to_sign(self, sign_angle, orb=5):
        """Check aspects between planets and the zodiac sign"""
        aspects = []
        
        aspect_types = {
            'Conjunction': 0,
            'Sextile': 60,
            'Square': 90,
            'Trine': 120,
            'Opposition': 180
        }
        
        for planet, planet_angle in self.planetary_positions.items():
            aspect_diff = self._calculate_aspect(sign_angle, planet_angle)
            
            for aspect_name, aspect_angle in aspect_types.items():
                if abs(aspect_diff - aspect_angle) <= orb:
                    aspects.append({
                        'planet': planet,
                        'aspect': aspect_name,
                        'orb': abs(aspect_diff - aspect_angle),
                        'strength': self._get_aspect_strength(aspect_name)
                    })
        
        return aspects
    
    def _get_aspect_strength(self, aspect_name):
        """Get strength rating for aspect type"""
        strengths = {
            'Conjunction': 5,
            'Opposition': 5,
            'Square': 4,
            'Trine': 3,
            'Sextile': 2
        }
        return strengths.get(aspect_name, 1)
    
    def _get_planet_speed(self, planet):
        """Get daily movement speed of planet in degrees"""
        speeds = {
            'Sun': 1.0,
            'Moon': 13.2,
            'Mercury': 1.2,
            'Venus': 1.1,
            'Mars': 0.5,
            'Jupiter': 0.08,
            'Saturn': 0.03,
            'Uranus': 0.04,
            'Neptune': 0.02,
            'Pluto': 0.03
        }
        return speeds.get(planet, 0.1)
    
    def _get_planet_influence(self, planet):
        """Get whether planet is bullish or bearish for the stock"""
        benefics = ['Venus', 'Jupiter']
        malefics = ['Mars', 'Saturn']
        
        if planet in benefics:
            return 'Bullish'
        elif planet in malefics:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def predict_trend(self):
        """Predict overall bullish/bearish trend for the stock"""
        if self.aspects:
            bullish_score = 0
            bearish_score = 0
            
            for aspect in self.aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                strength = aspect['strength']
                
                influence = self._get_planet_influence(planet)
                
                if influence == 'Bullish' and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                    bullish_score += strength
                elif influence == 'Bearish' and aspect_type in ['Conjunction', 'Opposition', 'Square']:
                    bearish_score += strength
                elif influence == 'Bullish' and aspect_type in ['Opposition', 'Square']:
                    bearish_score += strength * 0.5
                elif influence == 'Bearish' and aspect_type in ['Trine', 'Sextile']:
                    bullish_score += strength * 0.5
            
            if bullish_score > bearish_score:
                self.trend_prediction = {
                    'trend': 'Bullish',
                    'strength': min(5, (bullish_score - bearish_score) // 2 + 1),
                    'confidence': 'High' if (bullish_score - bearish_score) > 5 else 'Medium'
                }
            elif bearish_score > bullish_score:
                self.trend_prediction = {
                    'trend': 'Bearish',
                    'strength': min(5, (bearish_score - bullish_score) // 2 + 1),
                    'confidence': 'High' if (bearish_score - bullish_score) > 5 else 'Medium'
                }
            else:
                self.trend_prediction = {
                    'trend': 'Neutral',
                    'strength': 1,
                    'confidence': 'Low'
                }
        else:
            self.trend_prediction = {
                'trend': 'Neutral',
                'strength': 1,
                'confidence': 'Low'
            }
        
        return self.trend_prediction
    
    def predict_monthly_analysis(self):
        """Predict analysis for each month for the next 12 months"""
        # Get current date
        current_date = datetime.now()
        
        # Check each month for the next 12 months
        months_analysis = []
        
        for i in range(12):
            # Calculate the month to analyze
            target_date = current_date + timedelta(days=30 * i)
            target_month = target_date.month
            target_year = target_date.year
            
            # Calculate days difference
            days_diff = (target_date - current_date).days
            
            # Calculate planetary positions for that month
            future_positions = {}
            for planet, angle in self.planetary_positions.items():
                speed = self._get_planet_speed(planet)
                future_angle = (angle + speed * days_diff) % 360
                future_positions[planet] = future_angle
            
            # Check aspects to the stock's zodiac sign
            future_aspects = self._check_aspects_to_sign(self.zodiac_angle)
            
            # Calculate score for this month (ensure non-negative)
            month_score = 0
            for aspect in future_aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                strength = aspect['strength']
                
                influence = self._get_planet_influence(planet)
                
                if influence == 'Bullish' and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                    month_score += strength
                elif influence == 'Bearish' and aspect_type in ['Opposition', 'Square']:
                    month_score -= strength
            
            # Ensure score is non-negative
            month_score = max(0, month_score)
            
            # Determine key aspects for this month
            key_aspects = []
            for aspect in future_aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                strength = aspect['strength']
                influence = self._get_planet_influence(planet)
                
                key_aspects.append({
                    'planet': planet,
                    'aspect': aspect_type,
                    'influence': influence,
                    'strength': strength
                })
            
            months_analysis.append({
                'month': target_month,
                'year': target_year,
                'month_name': target_date.strftime('%B'),
                'score': month_score,
                'aspects': key_aspects,
                'date': target_date
            })
        
        self.monthly_analysis = months_analysis
        
        # Find the best month
        if months_analysis:
            best_month_data = max(months_analysis, key=lambda x: x['score'])
            self.best_month = {
                'month': best_month_data['month'],
                'year': best_month_data['year'],
                'month_name': best_month_data['month_name'],
                'score': best_month_data['score'],
                'aspects': best_month_data['aspects']
            }
        
        return months_analysis
    
    def predict_critical_dates(self):
        """Predict dates when the stock is expected to rise or fall"""
        # Get current date
        current_date = datetime.now()
        
        # Analyze the next 90 days
        critical_dates = []
        
        for i in range(90):
            target_date = current_date + timedelta(days=i)
            
            # Calculate days difference
            days_diff = i
            
            # Calculate planetary positions for that date
            future_positions = {}
            for planet, angle in self.planetary_positions.items():
                speed = self._get_planet_speed(planet)
                future_angle = (angle + speed * days_diff) % 360
                future_positions[planet] = future_angle
            
            # Check aspects to the stock's zodiac sign
            future_aspects = self._check_aspects_to_sign(self.zodiac_angle)
            
            # Calculate score for this date
            date_score = 0
            significant_aspects = []
            
            for aspect in future_aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                strength = aspect['strength']
                
                influence = self._get_planet_influence(planet)
                
                if influence == 'Bullish' and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                    date_score += strength
                    significant_aspects.append({
                        'planet': planet,
                        'aspect': aspect_type,
                        'influence': 'Bullish',
                        'strength': strength
                    })
                elif influence == 'Bearish' and aspect_type in ['Opposition', 'Square']:
                    date_score -= strength
                    significant_aspects.append({
                        'planet': planet,
                        'aspect': aspect_type,
                        'influence': 'Bearish',
                        'strength': strength
                    })
            
            # Determine if this is a critical date
            if abs(date_score) >= 3:  # Threshold for critical dates
                critical_dates.append({
                    'date': target_date,
                    'score': date_score,
                    'prediction': 'Rise' if date_score > 0 else 'Fall',
                    'significance': 'High' if abs(date_score) >= 5 else 'Medium',
                    'aspects': significant_aspects
                })
        
        self.critical_dates = critical_dates
        return critical_dates
    
    def predict_price_projections(self, current_price):
        """Predict price projections based on planetary aspects"""
        if not current_price:
            return None
            
        # Get current date
        current_date = datetime.now()
        
        # Analyze the next 365 days
        price_projections = []
        daily_prices = []
        daily_dates = []
        
        # Start with current price
        price = current_price
        daily_prices.append(price)
        daily_dates.append(current_date)
        
        # Calculate daily price changes
        for i in range(1, 365):
            target_date = current_date + timedelta(days=i)
            
            # Calculate days difference
            days_diff = i
            
            # Calculate planetary positions for that date
            future_positions = {}
            for planet, angle in self.planetary_positions.items():
                speed = self._get_planet_speed(planet)
                future_angle = (angle + speed * days_diff) % 360
                future_positions[planet] = future_angle
            
            # Check aspects to the stock's zodiac sign
            future_aspects = self._check_aspects_to_sign(self.zodiac_angle)
            
            # Calculate score for this date
            date_score = 0
            for aspect in future_aspects:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                strength = aspect['strength']
                
                influence = self._get_planet_influence(planet)
                
                if influence == 'Bullish' and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                    date_score += strength
                elif influence == 'Bearish' and aspect_type in ['Opposition', 'Square']:
                    date_score -= strength
            
            # Calculate price change based on score
            # Normalize score to a percentage change
            price_change_percent = date_score * 0.01  # 1% per point
            price_change = price * price_change_percent / 100
            
            # Apply price change with some randomness
            price = price * (1 + price_change_percent / 100 + np.random.normal(0, 0.005))
            daily_prices.append(price)
            daily_dates.append(target_date)
            
            # Record significant price movements
            if abs(price_change_percent) > 2:  # More than 2% change
                price_projections.append({
                    'date': target_date,
                    'price': price,
                    'change': price_change_percent,
                    'direction': 'Up' if price_change_percent > 0 else 'Down',
                    'score': date_score,
                    'aspects': future_aspects
                })
        
        self.price_projections = {
            'daily_prices': daily_prices,
            'daily_dates': daily_dates,
            'significant_movements': price_projections
        }
        
        return self.price_projections
    
    def create_planetary_chart(self):
        """Create a chart showing planetary positions and aspects to the stock's zodiac sign"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Zodiac signs
        zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        
        # Set up the chart
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 12)
        
        # Draw zodiac segments
        for i, sign in enumerate(zodiac_signs):
            angle = i * 30
            ax.text(angle, 11.5, sign, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.plot([angle, angle], [0, 12], 'k-', alpha=0.3, linewidth=1)
        
        # Plot planets
        planet_symbols = {
            'Sun': '☉', 'Moon': '☽', 'Mercury': '☿', 'Venus': '♀', 'Mars': '♂',
            'Jupiter': '♃', 'Saturn': '♄', 'Uranus': '♅', 'Neptune': '♆', 'Pluto': '♇'
        }
        
        planet_colors = {
            'Sun': 'gold', 'Moon': 'silver', 'Mercury': 'gray', 'Venus': 'lightgreen',
            'Mars': 'red', 'Jupiter': 'orange', 'Saturn': 'darkgoldenrod',
            'Uranus': 'lightblue', 'Neptune': 'darkblue', 'Pluto': 'darkred'
        }
        
        for planet, angle in self.planetary_positions.items():
            ax.plot([angle], [8], 'o', color=planet_colors[planet], markersize=12, alpha=0.8)
            ax.text(angle, 8.5, planet_symbols[planet], ha='center', va='center', 
                    fontsize=16, color=planet_colors[planet], fontweight='bold')
            ax.text(angle, 7.5, planet, ha='center', va='center', fontsize=10, color='black')
        
        # Plot stock's zodiac sign
        ax.plot([self.zodiac_angle], [5], 'o', color='purple', markersize=15)
        ax.text(self.zodiac_angle, 5.5, self.stock_name, ha='center', va='center', 
                fontsize=12, color='purple', fontweight='bold')
        ax.text(self.zodiac_angle, 4.5, self.zodiac_sign, ha='center', va='center', 
                fontsize=10, color='purple')
        
        # Draw aspects
        aspect_colors = {
            'Conjunction': 'black', 'Opposition': 'red', 'Trine': 'blue',
            'Square': 'green', 'Sextile': 'purple'
        }
        
        for aspect in self.aspects:
            planet = aspect['planet']
            aspect_type = aspect['aspect']
            planet_angle = self.planetary_positions[planet]
            
            ax.plot([self.zodiac_angle, planet_angle], [5, 8], 
                   color=aspect_colors[aspect_type], linewidth=2, alpha=0.7)
        
        # Add legend
        aspect_legend = []
        for aspect, color in aspect_colors.items():
            aspect_legend.append(plt.Line2D([0], [0], color=color, linewidth=2, label=aspect))
        
        ax.legend(handles=aspect_legend, loc='upper right', 
                 bbox_to_anchor=(1.3, 0.85), fontsize=10, title='Aspects')
        
        # Add title
        plt.title(f'{self.stock_name} ({self.zodiac_sign}) - Planetary Aspects\n'
                  f'Current Planetary Positions', fontsize=18, pad=30, fontweight='bold')
        
        # Remove grid and axis labels
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def create_price_projection_chart(self, current_price=None):
        """Create a chart showing price projections based on planetary aspects"""
        if not current_price or not self.price_projections:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get price data
        daily_prices = self.price_projections['daily_prices']
        daily_dates = self.price_projections['daily_dates']
        significant_movements = self.price_projections['significant_movements']
        
        # Plot price line
        ax.plot(daily_dates, daily_prices, 'b-', linewidth=2, label='Projected Price')
        
        # Plot significant movements
        if significant_movements:
            for movement in significant_movements:
                date = movement['date']
                price = movement['price']
                direction = movement['direction']
                
                if direction == 'Up':
                    ax.scatter(date, price, color='green', s=100, alpha=0.7, edgecolors='black')
                    ax.annotate(f"Buy\n{date.strftime('%Y-%m-%d')}", 
                               xy=(date, price), xytext=(0, 20),
                               textcoords='offset points', ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                else:
                    ax.scatter(date, price, color='red', s=100, alpha=0.7, edgecolors='black')
                    ax.annotate(f"Sell\n{date.strftime('%Y-%m-%d')}", 
                               xy=(date, price), xytext=(0, -30),
                               textcoords='offset points', ha='center', va='top',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Price Projection Based on Planetary Aspects', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis
        ax.yaxis.set_major_formatter('${x:,.2f}')
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_bullish_bearish_timeline(self):
        """Create a timeline showing bullish and bearish periods"""
        if not self.monthly_analysis:
            self.predict_monthly_analysis()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        dates = [data['date'] for data in self.monthly_analysis]
        scores = [data['score'] for data in self.monthly_analysis]
        
        # Create a timeline with color coding
        for i, (date, score) in enumerate(zip(dates, scores)):
            # Determine color based on score
            if score > 7:
                color = 'darkgreen'  # Strong Bullish
                label = 'Strong Bullish'
            elif score > 5:
                color = 'green'  # Bullish
                label = 'Bullish'
            elif score > 3:
                color = 'orange'  # Neutral
                label = 'Neutral'
            else:
                color = 'red'  # Bearish
                label = 'Bearish'
            
            # Draw bar for each month
            ax.barh(0, 30, left=date, height=0.5, color=color, alpha=0.7, label=label if i == 0 or i == len(scores)-1 or (i > 0 and label != [d['score'] for d in self.monthly_analysis][i-1]) else "")
            
            # Add month label
            ax.text(date + 15, 0, date.strftime('%b'), ha='center', va='center', fontsize=10)
        
        # Add critical dates as markers
        if self.critical_dates:
            for date_data in self.critical_dates:
                date = date_data['date']
                prediction = date_data['prediction']
                
                if prediction == 'Rise':
                    ax.scatter(date, 0, color='green', s=100, marker='^', zorder=5)
                else:
                    ax.scatter(date, 0, color='red', s=100, marker='v', zorder=5)
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Bullish/Bearish Timeline', fontsize=16, fontweight='bold')
        ax.set_yticks([])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def create_trading_signals_chart(self):
        """Create a chart showing when to go long or short"""
        if not self.critical_dates:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        dates = [data['date'] for data in self.critical_dates]
        scores = [data['score'] for data in self.critical_dates]
        predictions = [data['prediction'] for data in self.critical_dates]
        significances = [data['significance'] for data in self.critical_dates]
        
        # Create a scatter plot
        colors = ['green' if pred == 'Rise' else 'red' for pred in predictions]
        sizes = [150 if sig == 'High' else 100 for sig in significances]
        
        # Plot points
        for i, (date, score, pred, sig) in enumerate(zip(dates, scores, predictions, significances)):
            if pred == 'Rise':
                ax.scatter(date, score, color='green', s=sizes[i], alpha=0.7, edgecolors='black', marker='^')
                ax.annotate(f"GO LONG\n{date.strftime('%Y-%m-%d')}\n{sig} Significance", 
                           xy=(date, score), xytext=(0, 20),
                           textcoords='offset points', ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))
            else:
                ax.scatter(date, score, color='red', s=sizes[i], alpha=0.7, edgecolors='black', marker='v')
                ax.annotate(f"GO SHORT\n{date.strftime('%Y-%m-%d')}\n{sig} Significance", 
                           xy=(date, score), xytext=(0, -30),
                           textcoords='offset points', ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7))
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Trading Signals Based on Planetary Aspects', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Bullish/Bearish Score', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_coming_month_report(self):
        """Generate a detailed report for the coming month"""
        if not self.monthly_analysis:
            self.predict_monthly_analysis()
        
        # Get the next month (index 1 because index 0 is current month)
        if len(self.monthly_analysis) > 1:
            coming_month = self.monthly_analysis[1]
        else:
            coming_month = self.monthly_analysis[0]
        
        # Generate report
        report = f"""
# {coming_month['month_name']} {coming_month['year']} Performance Report

## Overall Score: {coming_month['score']:.1f}/10

### Performance Assessment:
"""
        
        if coming_month['score'] > 7:
            report += "Strong Bullish trend expected. This month shows excellent potential for growth.\n\n"
        elif coming_month['score'] > 5:
            report += "Moderate Bullish trend expected. Positive performance is likely.\n\n"
        elif coming_month['score'] > 3:
            report += "Neutral to slightly Bullish trend expected. Mixed performance possible.\n\n"
        else:
            report += "Bearish trend expected. Caution is advised for this month.\n\n"
        
        report += "### Key Planetary Aspects:\n\n"
        
        for aspect in coming_month['aspects']:
            influence = aspect['influence']
            planet = aspect['planet']
            aspect_type = aspect['aspect']
            strength = aspect['strength']
            
            report += f"- **{planet} {aspect_type}**: {influence} influence (Strength: {strength}/5)\n"
        
        report += "\n### Critical Dates in This Month:\n\n"
        
        if self.critical_dates:
            month_critical_dates = [
                date for date in self.critical_dates 
                if date['date'].month == coming_month['month'] and 
                   date['date'].year == coming_month['year']
            ]
            
            if month_critical_dates:
                for date_data in month_critical_dates:
                    date = date_data['date']
                    prediction = date_data['prediction']
                    significance = date_data['significance']
                    
                    if prediction == 'Rise':
                        report += f"- **{date.strftime('%Y-%m-%d')}**: Expected Rise ({significance} significance) - Consider going LONG before this date\n"
                    else:
                        report += f"- **{date.strftime('%Y-%m-%d')}**: Expected Fall ({significance} significance) - Consider going SHORT before this date\n"
            else:
                report += "No critical dates identified for this month.\n"
        else:
            report += "No critical dates identified for this month.\n"
        
        report += "\n### Trading Recommendations:\n\n"
        
        if coming_month['score'] > 5:
            report += "- Consider long positions during favorable aspects\n"
            report += "- Monitor for entry points around critical dates marked as 'Rise'\n"
            report += "- Set stop-losses to protect against unexpected reversals\n"
        else:
            report += "- Consider short positions or hedging strategies\n"
            report += "- Avoid new long positions during unfavorable aspects\n"
            report += "- Be cautious around critical dates marked as 'Fall'\n"
        
        report += "\n### Upcoming Bullish/Bearish Moves:\n\n"
        
        # Find upcoming critical dates in the next 30 days
        upcoming_dates = [
            date for date in self.critical_dates 
            if date['date'] <= datetime.now() + timedelta(days=30)
        ]
        
        if upcoming_dates:
            for date_data in upcoming_dates:
                date = date_data['date']
                prediction = date_data['prediction']
                significance = date_data['significance']
                
                if prediction == 'Rise':
                    report += f"- **{date.strftime('%Y-%m-%d')}**: Bullish move expected ({significance} significance)\n"
                else:
                    report += f"- **{date.strftime('%Y-%m-%d')}**: Bearish move expected ({significance} significance)\n"
        else:
            report += "No significant bullish/bearish moves expected in the next 30 days.\n"
        
        report += "\n*This report is generated based on astrological analysis and should be used in conjunction with other market analysis.*"
        
        return report

# ======================
# STREAMLIT APP
# ======================
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Stock Zodiac Analysis",
        page_icon="♈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #43A047;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .chart-container {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .report-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .price-input {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>♈ Stock Zodiac Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Predict stock trends based on zodiac signs and planetary aspects</p>", unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("Stock Information")
    
    # Stock name input
    stock_name = st.sidebar.text_input("Stock Name", value="AAPL")
    
    # Current price input (optional)
    current_price = st.sidebar.number_input("Current Stock Price ($)", min_value=0.01, value=150.0, step=0.01)
    
    # Zodiac sign selection (optional)
    zodiac_signs = ['Auto-detect', 'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    selected_zodiac = st.sidebar.selectbox("Zodiac Sign (Optional)", zodiac_signs, index=0)
    
    # Convert "Auto-detect" to None for processing
    if selected_zodiac == "Auto-detect":
        selected_zodiac = None
    
    # Incorporation date (optional)
    incorporation_date = st.sidebar.date_input("Incorporation Date (Optional)", 
                                             datetime.now().date() - timedelta(days=365*5))
    
    # Analyze button
    analyze_button = st.sidebar.button("Analyze Stock", key="analyze")
    
    # Main content area
    if analyze_button:
        # Create analysis object
        analysis = StockZodiacAnalysis(
            stock_name=stock_name,
            zodiac_sign=selected_zodiac,
            incorporation_date=incorporation_date
        )
        
        # Display stock information
        st.header(f"{stock_name} Analysis")
        st.subheader(f"Zodiac Sign: {analysis.zodiac_sign}")
        
        # Display how zodiac sign was determined
        if selected_zodiac is None:
            st.info(f"Zodiac sign automatically determined based on stock name: {stock_name}")
        
        # Display planetary positions
        st.subheader("Current Planetary Positions")
        planet_data = []
        for planet, angle in analysis.planetary_positions.items():
            zodiac = analysis._get_zodiac_from_angle(angle)
            planet_data.append({"Planet": planet, "Angle": f"{angle:.2f}°", "Zodiac": zodiac})
        
        planet_df = pd.DataFrame(planet_data)
        st.dataframe(planet_df, use_container_width=True)
        
        # Display aspects
        st.subheader("Aspects to Stock's Zodiac Sign")
        if analysis.aspects:
            aspect_data = []
            for aspect in analysis.aspects:
                aspect_data.append({
                    "Planet": aspect['planet'],
                    "Aspect": aspect['aspect'],
                    "Orb": f"{aspect['orb']:.2f}°",
                    "Strength": aspect['strength']
                })
            
            aspect_df = pd.DataFrame(aspect_data)
            st.dataframe(aspect_df, use_container_width=True)
        else:
            st.info("No significant aspects found")
        
        # Predict and display trend
        st.subheader("Trend Prediction")
        trend = analysis.predict_trend()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Overall Trend")
            if trend['trend'] == 'Bullish':
                st.success(f"**Bullish** (Strength: {trend['strength']}/5)")
            elif trend['trend'] == 'Bearish':
                st.error(f"**Bearish** (Strength: {trend['strength']}/5)")
            else:
                st.info(f"**Neutral** (Strength: {trend['strength']}/5)")
            
            st.markdown(f"**Confidence**: {trend['confidence']}")
        
        with col2:
            st.markdown("#### Key Influences")
            if analysis.aspects:
                for aspect in analysis.aspects:
                    planet = aspect['planet']
                    aspect_type = aspect['aspect']
                    influence = analysis._get_planet_influence(planet)
                    
                    if influence == 'Bullish':
                        st.success(f"{planet} {aspect_type}: Bullish influence")
                    elif influence == 'Bearish':
                        st.error(f"{planet} {aspect_type}: Bearish influence")
                    else:
                        st.info(f"{planet} {aspect_type}: Neutral influence")
            else:
                st.info("No significant planetary influences")
        
        # Predict and display monthly analysis
        st.subheader("Monthly Performance Analysis")
        monthly_analysis = analysis.predict_monthly_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Best Performing Month")
            if analysis.best_month:
                st.success(f"**{analysis.best_month['month_name']} {analysis.best_month['year']}**")
                st.markdown(f"**Score**: {analysis.best_month['score']:.1f}")
                
                st.markdown("#### Key Aspects in Best Month")
                for aspect in analysis.best_month['aspects']:
                    planet = aspect['planet']
                    aspect_type = aspect['aspect']
                    influence = aspect['influence']
                    
                    if influence == 'Bullish':
                        st.success(f"{planet} {aspect_type}: Bullish")
                    elif influence == 'Bearish':
                        st.error(f"{planet} {aspect_type}: Bearish")
                    else:
                        st.info(f"{planet} {aspect_type}: Neutral")
            else:
                st.info("No best month identified")
        
        with col2:
            st.markdown("#### All Months Performance")
            # Create a DataFrame for display
            months_data = []
            for month_data in monthly_analysis:
                months_data.append({
                    'Month': f"{month_data['month_name']} {month_data['year']}",
                    'Score': month_data['score'],
                    'Performance': 'Excellent' if month_data['score'] > 7 else 
                                  'Good' if month_data['score'] > 5 else 
                                  'Moderate' if month_data['score'] > 3 else 'Poor'
                })
            
            months_df = pd.DataFrame(months_data)
            st.dataframe(months_df, use_container_width=True)
        
        # Predict and display critical dates
        st.subheader("Critical Dates Prediction")
        critical_dates = analysis.predict_critical_dates()
        
        if critical_dates:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Upcoming Critical Dates")
                for date_data in critical_dates[:5]:  # Show next 5 critical dates
                    date = date_data['date']
                    prediction = date_data['prediction']
                    significance = date_data['significance']
                    
                    if prediction == 'Rise':
                        st.success(f"**{date.strftime('%Y-%m-%d')}**: Expected Rise ({significance})")
                    else:
                        st.error(f"**{date.strftime('%Y-%m-%d')}**: Expected Fall ({significance})")
            
            with col2:
                st.markdown("#### All Critical Dates")
                critical_df = pd.DataFrame(critical_dates)
                critical_df['date'] = critical_df['date'].dt.strftime('%Y-%m-%d')
                critical_df = critical_df[['date', 'prediction', 'significance', 'score']]
                st.dataframe(critical_df, use_container_width=True)
        else:
            st.info("No critical dates predicted for the next 90 days")
        
        # Display charts
        st.subheader("Visualizations")
        
        # Price projection chart (if current price is provided)
        if current_price:
            st.markdown("#### Price Projection Chart")
            price_projections = analysis.predict_price_projections(current_price)
            fig_price = analysis.create_price_projection_chart(current_price)
            if fig_price:
                st.pyplot(fig_price)
        
        # Bullish/Bearish timeline
        st.markdown("#### Bullish/Bearish Timeline")
        fig_timeline = analysis.create_bullish_bearish_timeline()
        st.pyplot(fig_timeline)
        
        # Trading signals chart
        st.markdown("#### Trading Signals")
        fig_signals = analysis.create_trading_signals_chart()
        if fig_signals:
            st.pyplot(fig_signals)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Planetary Aspects Chart")
            fig1 = analysis.create_planetary_chart()
            st.pyplot(fig1)
        
        with col2:
            st.markdown("#### Critical Dates Chart")
            if critical_dates:
                fig3 = analysis.create_critical_dates_chart()
                if fig3:
                    st.pyplot(fig3)
        
        # Coming month report
        st.subheader("Coming Month Performance Report")
        report = analysis.generate_coming_month_report()
        st.markdown(report, unsafe_allow_html=True)
        
        # Add disclaimer
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for educational purposes only. Astrological analysis should not be used as the sole basis for trading decisions. Always conduct thorough research and consider multiple factors before making financial decisions.")

if __name__ == "__main__":
    main()
