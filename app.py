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
from matplotlib.patches import Rectangle

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
        # Create figure with proper size
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
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
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_price_projection_chart(self, current_price=None):
        """Create a chart showing price projections based on planetary aspects"""
        if not current_price or not self.price_projections:
            return None
        
        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Get price data
        daily_prices = self.price_projections['daily_prices']
        daily_dates = self.price_projections['daily_dates']
        significant_movements = self.price_projections['significant_movements']
        
        # Plot price line with increased width for better visibility
        ax.plot(daily_dates, daily_prices, 'b-', linewidth=3, label='Projected Price')
        
        # Plot significant movements with larger markers
        if significant_movements:
            for movement in significant_movements:
                date = movement['date']
                price = movement['price']
                direction = movement['direction']
                
                if direction == 'Up':
                    ax.scatter(date, price, color='green', s=150, alpha=0.8, edgecolors='black', linewidth=2)
                    ax.annotate(f"Buy\n{date.strftime('%Y-%m-%d')}", 
                               xy=(date, price), xytext=(0, 20),
                               textcoords='offset points', ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                else:
                    ax.scatter(date, price, color='red', s=150, alpha=0.8, edgecolors='black', linewidth=2)
                    ax.annotate(f"Sell\n{date.strftime('%Y-%m-%d')}", 
                               xy=(date, price), xytext=(0, -30),
                               textcoords='offset points', ha='center', va='top',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Price Projection Based on Planetary Aspects', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
        
        # Format y-axis
        ax.yaxis.set_major_formatter('${x:,.2f}')
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
        
        # Add legend with larger font
        ax.legend(loc='upper left', fontsize=12)
        
        # Add grid with better visibility
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_bullish_bearish_timeline(self):
        """Create a timeline showing bullish and bearish periods"""
        if not self.monthly_analysis:
            self.predict_monthly_analysis()
        
        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
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
            
            # Draw bar for each month with increased height for better visibility
            ax.barh(0, 30, left=date, height=0.8, color=color, alpha=0.8, 
                   label=label if i == 0 or i == len(scores)-1 or (i > 0 and label != [d['score'] for d in self.monthly_analysis][i-1]) else "")
            
            # Add month label with larger font
            ax.text(date + timedelta(days=15), 0, date.strftime('%b'), ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add critical dates as markers with larger size
        if self.critical_dates:
            for date_data in self.critical_dates:
                date = date_data['date']
                prediction = date_data['prediction']
                
                if prediction == 'Rise':
                    ax.scatter(date, 0, color='green', s=150, marker='^', zorder=5, edgecolors='black', linewidth=2)
                else:
                    ax.scatter(date, 0, color='red', s=150, marker='v', zorder=5, edgecolors='black', linewidth=2)
        
        # Add key aspect transit dates with larger markers
        if self.monthly_analysis:
            for month_data in self.monthly_analysis:
                month_date = month_data['date']
                aspects = month_data['aspects']
                
                for aspect in aspects:
                    # Create a transit date (middle of the month)
                    transit_date = month_date + timedelta(days=15)
                    
                    # Determine color based on influence
                    if aspect['influence'] == 'Bullish':
                        color = 'lightgreen'
                    elif aspect['influence'] == 'Bearish':
                        color = 'lightcoral'
                    else:
                        color = 'lightgray'
                    
                    # Add marker for transit with larger size
                    ax.scatter(transit_date, 0.4, color=color, s=100, alpha=0.8, zorder=4)
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Bullish/Bearish Timeline with Key Aspect Transits', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_yticks([])
        
        # Format x-axis with larger font
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
        
        # Add legend with larger font
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
        
        # Add grid with better visibility
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_trading_signals_chart(self):
        """Create a chart showing when to go long or short"""
        if not self.critical_dates:
            return None
        
        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Prepare data
        dates = [data['date'] for data in self.critical_dates]
        scores = [data['score'] for data in self.critical_dates]
        predictions = [data['prediction'] for data in self.critical_dates]
        significances = [data['significance'] for data in self.critical_dates]
        
        # Create a scatter plot with larger markers
        colors = ['green' if pred == 'Rise' else 'red' for pred in predictions]
        sizes = [200 if sig == 'High' else 150 for sig in significances]
        
        # Plot points with larger size and better visibility
        for i, (date, score, pred, sig) in enumerate(zip(dates, scores, predictions, significances)):
            if pred == 'Rise':
                ax.scatter(date, score, color='green', s=sizes[i], alpha=0.8, edgecolors='black', linewidth=2, marker='^')
                ax.annotate(f"GO LONG\n{date.strftime('%Y-%m-%d')}\n{sig} Significance", 
                           xy=(date, score), xytext=(0, 30),
                           textcoords='offset points', ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=12, fontweight='bold')
            else:
                ax.scatter(date, score, color='red', s=sizes[i], alpha=0.8, edgecolors='black', linewidth=2, marker='v')
                ax.annotate(f"GO SHORT\n{date.strftime('%Y-%m-%d')}\n{sig} Significance", 
                           xy=(date, score), xytext=(0, -40),
                           textcoords='offset points', ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=12, fontweight='bold')
        
        # Add zero line with better visibility
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Trading Signals Based on Planetary Aspects', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Bullish/Bearish Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        
        # Format x-axis with larger font
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
        
        # Format y-axis with larger font
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
        
        # Add grid with better visibility
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_critical_dates_chart(self):
        """Create a chart showing critical dates for the stock"""
        if not self.critical_dates:
            return None
        
        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Prepare data
        dates = [data['date'] for data in self.critical_dates]
        scores = [data['score'] for data in self.critical_dates]
        predictions = [data['prediction'] for data in self.critical_dates]
        significances = [data['significance'] for data in self.critical_dates]
        
        # Create color map based on prediction
        colors = ['green' if pred == 'Rise' else 'red' for pred in predictions]
        
        # Create size map based on significance
        sizes = [150 if sig == 'High' else 100 for sig in significances]
        
        # Plot scatter with larger markers
        scatter = ax.scatter(dates, scores, c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add zero line with better visibility
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        
        # Add labels for each point with larger font
        for i, (date, score, pred, sig) in enumerate(zip(dates, scores, predictions, significances)):
            ax.annotate(f"{pred}\n{sig}", xy=(date, score), xytext=(0, 20),
                        textcoords='offset points', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                        fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Critical Dates Prediction', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Bullish/Bearish Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        
        # Format x-axis with larger font
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
        
        # Format y-axis with larger font
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
        
        # Add legend with larger font
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='Rise'),
            Patch(facecolor='red', alpha=0.8, label='Fall'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=12, label='Medium Significance', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=16, label='High Significance', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Add grid with better visibility
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_monthly_detail_chart(self, month_index, current_price=None):
        """Create a detailed chart for a specific month"""
        if not self.monthly_analysis or month_index >= len(self.monthly_analysis):
            return None
        
        # Get the selected month data
        month_data = self.monthly_analysis[month_index]
        month_date = month_data['date']
        month_name = month_data['month_name']
        year = month_data['year']
        score = month_data['score']
        aspects = month_data['aspects']
        
        # Create figure with proper size
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 2, height_ratios=[1, 2, 2], hspace=0.3)
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        
        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, f'{month_name} {year} - Detailed Analysis', 
                     ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Score and performance
        ax_score = fig.add_subplot(gs[1, 0])
        ax_score.axis('off')
        ax_score.set_facecolor('white')
        
        # Determine performance category
        if score > 7:
            performance = "Excellent"
            color = 'darkgreen'
        elif score > 5:
            performance = "Good"
            color = 'green'
        elif score > 3:
            performance = "Moderate"
            color = 'orange'
        else:
            performance = "Poor"
            color = 'red'
        
        # Display score with larger font
        ax_score.text(0.5, 0.7, f'Score: {score:.1f}/10', ha='center', va='center', fontsize=28, fontweight='bold')
        ax_score.text(0.5, 0.4, f'Performance: {performance}', ha='center', va='center', fontsize=20, color=color, fontweight='bold')
        
        # Key aspects
        ax_aspects = fig.add_subplot(gs[1, 1])
        ax_aspects.axis('off')
        ax_aspects.set_facecolor('white')
        
        # Display aspects with larger font
        ax_aspects.text(0.5, 0.9, 'Key Planetary Aspects', ha='center', va='center', fontsize=18, fontweight='bold')
        
        y_pos = 0.7
        for aspect in aspects:
            planet = aspect['planet']
            aspect_type = aspect['aspect']
            influence = aspect['influence']
            strength = aspect['strength']
            
            if influence == 'Bullish':
                color = 'green'
            elif influence == 'Bearish':
                color = 'red'
            else:
                color = 'gray'
            
            ax_aspects.text(0.5, y_pos, f'{planet} {aspect_type}: {influence} (Strength: {strength}/5)', 
                           ha='center', va='center', fontsize=14, color=color, fontweight='bold')
            y_pos -= 0.18
        
        # Price projection for the month
        ax_price = fig.add_subplot(gs[2, :])
        ax_price.set_facecolor('white')
        
        if current_price and self.price_projections:
            # Get the price data for this month
            start_date = month_date
            end_date = month_date + timedelta(days=30)
            
            # Filter price data for this month
            daily_dates = self.price_projections['daily_dates']
            daily_prices = self.price_projections['daily_prices']
            
            month_dates = [d for d in daily_dates if start_date <= d <= end_date]
            month_prices = [daily_prices[daily_dates.index(d)] for d in month_dates]
            
            # Plot price line with increased width
            ax_price.plot(month_dates, month_prices, 'b-', linewidth=3, label='Projected Price')
            
            # Add critical dates for this month with larger markers
            if self.critical_dates:
                month_critical_dates = [
                    date for date in self.critical_dates 
                    if start_date <= date['date'] <= end_date
                ]
                
                for date_data in month_critical_dates:
                    date = date_data['date']
                    prediction = date_data['prediction']
                    
                    # Find the closest date in month_dates to the critical date
                    closest_date = min(month_dates, key=lambda d: abs(d - date))
                    closest_index = month_dates.index(closest_date)
                    
                    if prediction == 'Rise':
                        ax_price.scatter(closest_date, month_prices[closest_index], color='green', s=150, alpha=0.8, edgecolors='black', linewidth=2)
                        ax_price.annotate(f"Buy\n{date.strftime('%d')}", 
                                       xy=(closest_date, month_prices[closest_index]), xytext=(0, 20),
                                       textcoords='offset points', ha='center', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                                       fontsize=12, fontweight='bold')
                    else:
                        ax_price.scatter(closest_date, month_prices[closest_index], color='red', s=150, alpha=0.8, edgecolors='black', linewidth=2)
                        ax_price.annotate(f"Sell\n{date.strftime('%d')}", 
                                       xy=(closest_date, month_prices[closest_index]), xytext=(0, -30),
                                       textcoords='offset points', ha='center', va='top',
                                       bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                                       fontsize=12, fontweight='bold')
            
            # Formatting
            ax_price.set_title(f'Price Projection for {month_name} {year}', fontsize=16, fontweight='bold', pad=20)
            ax_price.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
            ax_price.set_xlabel('Date', fontsize=14, fontweight='bold')
            
            # Format x-axis with larger font
            ax_price.xaxis.set_major_formatter(DateFormatter('%d'))
            plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
            
            # Format y-axis with larger font
            ax_price.yaxis.set_major_formatter('${x:,.2f}')
            plt.setp(ax_price.yaxis.get_majorticklabels(), fontsize=12)
            
            # Add legend with larger font
            ax_price.legend(loc='upper left', fontsize=12)
            
            # Add grid with better visibility
            ax_price.grid(True, alpha=0.3, linestyle='--')
        else:
            ax_price.axis('off')
            ax_price.text(0.5, 0.5, 'Price projection not available. Please provide current stock price.', 
                        ha='center', va='center', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_monthly_swing_chart(self, month_index, current_price=None):
        """Create a chart showing bullish/bearish swings with high and low points for a specific month"""
        if not self.monthly_analysis or month_index >= len(self.monthly_analysis):
            return None
        
        # Get the selected month data
        month_data = self.monthly_analysis[month_index]
        month_date = month_data['date']
        month_name = month_data['month_name']
        year = month_data['year']
        score = month_data['score']
        
        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Set background color for better visibility
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Get the price data for this month
        if current_price and self.price_projections:
            start_date = month_date
            end_date = month_date + timedelta(days=30)
            
            # Filter price data for this month
            daily_dates = self.price_projections['daily_dates']
            daily_prices = self.price_projections['daily_prices']
            
            month_dates = [d for d in daily_dates if start_date <= d <= end_date]
            month_prices = [daily_prices[daily_dates.index(d)] for d in month_dates]
            
            # Plot price line with increased width
            ax.plot(month_dates, month_prices, 'b-', linewidth=3, label='Price Movement')
            
            # Identify high and low points
            high_points = []
            low_points = []
            
            # Simple algorithm to find local maxima and minima
            for i in range(1, len(month_prices) - 1):
                if month_prices[i] > month_prices[i-1] and month_prices[i] > month_prices[i+1]:
                    high_points.append((month_dates[i], month_prices[i]))
                elif month_prices[i] < month_prices[i-1] and month_prices[i] < month_prices[i+1]:
                    low_points.append((month_dates[i], month_prices[i]))
            
            # Plot high points
            for date, price in high_points:
                ax.scatter(date, price, color='green', s=150, marker='^', zorder=5, edgecolors='black', linewidth=2)
                ax.annotate(f"High\n${price:.2f}", 
                           xy=(date, price), xytext=(0, 20),
                           textcoords='offset points', ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                           fontsize=12, fontweight='bold')
            
            # Plot low points
            for date, price in low_points:
                ax.scatter(date, price, color='red', s=150, marker='v', zorder=5, edgecolors='black', linewidth=2)
                ax.annotate(f"Low\n${price:.2f}", 
                           xy=(date, price), xytext=(0, -30),
                           textcoords='offset points', ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                           fontsize=12, fontweight='bold')
            
            # Add bullish/bearish swing indicators
            for i in range(1, len(month_prices)):
                if month_prices[i] > month_prices[i-1]:
                    # Bullish swing
                    ax.add_patch(Rectangle((month_dates[i-1], min(month_prices[i-1], month_prices[i])), 
                                        (month_dates[i] - month_dates[i-1]).days, 
                                        abs(month_prices[i] - month_prices[i-1]),
                                        alpha=0.2, color='green'))
                else:
                    # Bearish swing
                    ax.add_patch(Rectangle((month_dates[i-1], min(month_prices[i-1], month_prices[i])), 
                                        (month_dates[i] - month_dates[i-1]).days, 
                                        abs(month_prices[i] - month_prices[i-1]),
                                        alpha=0.2, color='red'))
            
            # Add critical dates for this month
            if self.critical_dates:
                month_critical_dates = [
                    date for date in self.critical_dates 
                    if start_date <= date['date'] <= end_date
                ]
                
                for date_data in month_critical_dates:
                    date = date_data['date']
                    prediction = date_data['prediction']
                    
                    # Find the closest date in month_dates to the critical date
                    closest_date = min(month_dates, key=lambda d: abs(d - date))
                    closest_index = month_dates.index(closest_date)
                    
                    if prediction == 'Rise':
                        ax.axvline(x=closest_date, color='green', linestyle='--', alpha=0.7, linewidth=2)
                        ax.text(closest_date, max(month_prices) * 0.95, "Bullish\nSignal", 
                               ha='center', va='top', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8))
                    else:
                        ax.axvline(x=closest_date, color='red', linestyle='--', alpha=0.7, linewidth=2)
                        ax.text(closest_date, max(month_prices) * 0.95, "Bearish\nSignal", 
                               ha='center', va='top', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8))
            
            # Formatting
            ax.set_title(f'{month_name} {year} - Bullish/Bearish Swings with High/Low Points', 
                         fontsize=18, fontweight='bold', pad=20)
            ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            
            # Format x-axis with larger font
            ax.xaxis.set_major_formatter(DateFormatter('%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
            
            # Format y-axis with larger font
            ax.yaxis.set_major_formatter('${x:,.2f}')
            plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
            
            # Add legend with larger font
            ax.legend(loc='upper left', fontsize=12)
            
            # Add grid with better visibility
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add overall trend indicator
            trend_color = 'green' if score > 5 else 'red'
            trend_text = 'Bullish Month' if score > 5 else 'Bearish Month'
            ax.text(0.02, 0.95, f"Overall Trend: {trend_text}", transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color=trend_color,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        else:
            # If no price data available, show a message
            ax.axis('off')
            ax.text(0.5, 0.5, 'Price projection not available. Please provide current stock price.', 
                    ha='center', va='center', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def generate_monthly_report(self, month_index):
        """Generate a detailed report for a specific month"""
        if not self.monthly_analysis or month_index >= len(self.monthly_analysis):
            return "No data available for the selected month."
        
        # Get the selected month data
        month_data = self.monthly_analysis[month_index]
        month_date = month_data['date']
        month_name = month_data['month_name']
        year = month_data['year']
        score = month_data['score']
        aspects = month_data['aspects']
        
        # Generate report
        report = f"""
# {month_name} {year} Performance Report

## Overall Score: {score:.1f}/10

### Performance Assessment:
"""
        
        if score > 7:
            report += "Strong Bullish trend expected. This month shows excellent potential for growth.\n\n"
        elif score > 5:
            report += "Moderate Bullish trend expected. Positive performance is likely.\n\n"
        elif score > 3:
            report += "Neutral to slightly Bullish trend expected. Mixed performance possible.\n\n"
        else:
            report += "Bearish trend expected. Caution is advised for this month.\n\n"
        
        report += "### Key Planetary Aspects:\n\n"
        
        for aspect in aspects:
            influence = aspect['influence']
            planet = aspect['planet']
            aspect_type = aspect['aspect']
            strength = aspect['strength']
            
            report += f"- **{planet} {aspect_type}**: {influence} influence (Strength: {strength}/5)\n"
        
        report += "\n### Critical Dates in This Month:\n\n"
        
        if self.critical_dates:
            month_critical_dates = [
                date for date in self.critical_dates 
                if date['date'].month == month_date.month and 
                   date['date'].year == month_date.year
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
        
        if score > 5:
            report += "- Consider long positions during favorable aspects\n"
            report += "- Monitor for entry points around critical dates marked as 'Rise'\n"
            report += "- Set stop-losses to protect against unexpected reversals\n"
        else:
            report += "- Consider short positions or hedging strategies\n"
            report += "- Avoid new long positions during unfavorable aspects\n"
            report += "- Be cautious around critical dates marked as 'Fall'\n"
        
        report += "\n*This report is generated based on astrological analysis and should be used in conjunction with other market analysis.*"
        
        return report
    
    def generate_coming_month_report(self):
        """Generate a detailed report for the coming month"""
        if not self.monthly_analysis:
            self.predict_monthly_analysis()
        
        # Get the next month (index 1 because index 0 is current month)
        if len(self.monthly_analysis) > 1:
            coming_month_index = 1
        else:
            coming_month_index = 0
        
        return self.generate_monthly_report(coming_month_index)

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
    .month-selector {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .publish-section {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
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
                plt.close(fig_price)  # Close figure to free memory
        
        # Bullish/Bearish timeline
        st.markdown("#### Bullish/Bearish Timeline with Key Aspect Transits")
        fig_timeline = analysis.create_bullish_bearish_timeline()
        st.pyplot(fig_timeline)
        plt.close(fig_timeline)  # Close figure to free memory
        
        # Trading signals chart
        st.markdown("#### Trading Signals")
        fig_signals = analysis.create_trading_signals_chart()
        if fig_signals:
            st.pyplot(fig_signals)
            plt.close(fig_signals)  # Close figure to free memory
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Planetary Aspects Chart")
            fig1 = analysis.create_planetary_chart()
            st.pyplot(fig1)
            plt.close(fig1)  # Close figure to free memory
        
        with col2:
            st.markdown("#### Critical Dates Chart")
            if critical_dates:
                fig3 = analysis.create_critical_dates_chart()
                if fig3:
                    st.pyplot(fig3)
                    plt.close(fig3)  # Close figure to free memory
        
        # Month selector for detailed analysis
        st.subheader("Monthly Detailed Analysis")
        
        # Create month selector
        month_options = [f"{month_data['month_name']} {month_data['year']}" for month_data in monthly_analysis]
        selected_month = st.selectbox("Select a month for detailed analysis", month_options)
        
        if selected_month:
            # Get the index of the selected month
            month_index = month_options.index(selected_month)
            
            # Display detailed chart for the selected month
            st.markdown(f"#### {selected_month} - Detailed Analysis")
            fig_monthly = analysis.create_monthly_detail_chart(month_index, current_price)
            if fig_monthly:
                st.pyplot(fig_monthly)
                plt.close(fig_monthly)  # Close figure to free memory
            
            # Display the new monthly swing chart
            st.markdown(f"#### {selected_month} - Bullish/Bearish Swings with High/Low Points")
            fig_swing = analysis.create_monthly_swing_chart(month_index, current_price)
            if fig_swing:
                st.pyplot(fig_swing)
                plt.close(fig_swing)  # Close figure to free memory
            
            # Display detailed report for the selected month
            with st.expander(f"View Full Report for {selected_month}"):
                report = analysis.generate_monthly_report(month_index)
                st.markdown(report, unsafe_allow_html=True)
        
        # Month report publisher
        st.subheader("Publish Monthly Report")
        st.markdown("#### Select a month to publish its report")
        
        # Create month selector for publishing
        publish_month = st.selectbox("Select a month to publish", month_options, key="publish_month")
        
        if publish_month:
            # Get the index of the selected month
            publish_month_index = month_options.index(publish_month)
            
            # Display publish button
            if st.button(f"Publish Report for {publish_month}", key="publish_button"):
                # Generate the report
                report = analysis.generate_monthly_report(publish_month_index)
                
                # Display success message
                st.success(f"Report for {publish_month} has been published successfully!")
                
                # Display the published report
                st.markdown("### Published Report:")
                st.markdown(report, unsafe_allow_html=True)
        
        # Coming month report
        st.subheader("Coming Month Performance Report")
        report = analysis.generate_coming_month_report()
        st.markdown(report, unsafe_allow_html=True)
        
        # Add disclaimer
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for educational purposes only. Astrological analysis should not be used as the sole basis for trading decisions. Always conduct thorough research and consider multiple factors before making financial decisions.")

if __name__ == "__main__":
    main()
