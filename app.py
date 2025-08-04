# app.py - Fixed Stock Zodiac Analysis Program
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from datetime import datetime, timedelta, date
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates

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
    
    def predict_best_month(self):
        """Predict the best month for the stock based on planetary transits"""
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
            
            # Calculate score for this month
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
            
            months_analysis.append({
                'month': target_month,
                'year': target_year,
                'month_name': target_date.strftime('%B'),
                'score': month_score,
                'aspects': future_aspects
            })
        
        # Find the month with the highest score
        best_month = max(months_analysis, key=lambda x: x['score'])
        
        self.best_month = {
            'month': best_month['month'],
            'year': best_month['year'],
            'month_name': best_month['month_name'],
            'score': best_month['score'],
            'aspects': best_month['aspects']
        }
        
        return self.best_month
    
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
        return self.critical_dates
    
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
    
    def create_trend_chart(self):
        """Create a chart showing the predicted trend for the next 12 months"""
        # Get current date
        current_date = datetime.now()
        
        # Calculate monthly scores for the next 12 months
        months_data = []
        for i in range(12):
            target_date = current_date + timedelta(days=30 * i)
            days_diff = (target_date - current_date).days
            
            # Calculate planetary positions for that month
            future_positions = {}
            for planet, angle in self.planetary_positions.items():
                speed = self._get_planet_speed(planet)
                future_angle = (angle + speed * days_diff) % 360
                future_positions[planet] = future_angle
            
            # Check aspects to the stock's zodiac sign
            future_aspects = self._check_aspects_to_sign(self.zodiac_angle)
            
            # Calculate score for this month
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
            
            months_data.append({
                'date': target_date,
                'month_name': target_date.strftime('%B %Y'),
                'score': month_score
            })
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot scores
        dates = [data['date'] for data in months_data]
        scores = [data['score'] for data in months_data]
        month_names = [data['month_name'] for data in months_data]
        
        # Create color map based on score
        colors = ['red' if score < 0 else 'green' for score in scores]
        
        bars = ax.bar(dates, scores, color=colors, alpha=0.7)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Predicted Trend for Next 12 Months', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Bullish/Bearish Score', fontsize=12)
        ax.set_xlabel('Month', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}', ha='center', va='bottom' if height < 0 else 'top')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Bullish'),
            Patch(facecolor='red', alpha=0.7, label='Bearish')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Highlight best month
        if self.best_month:
            best_date = datetime(self.best_month['year'], self.best_month['month'], 15)
            for i, data in enumerate(months_data):
                if data['date'].month == self.best_month['month'] and data['date'].year == self.best_month['year']:
                    ax.annotate('Best Month', xy=(dates[i], scores[i]), xytext=(0, 20),
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                    break
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_critical_dates_chart(self):
        """Create a chart showing critical dates for the stock"""
        if not self.critical_dates:
            return None
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        dates = [data['date'] for data in self.critical_dates]
        scores = [data['score'] for data in self.critical_dates]
        predictions = [data['prediction'] for data in self.critical_dates]
        significances = [data['significance'] for data in self.critical_dates]
        
        # Create color map based on prediction
        colors = ['green' if pred == 'Rise' else 'red' for pred in predictions]
        
        # Create size map based on significance
        sizes = [100 if sig == 'High' else 50 for sig in significances]
        
        # Plot scatter
        scatter = ax.scatter(dates, scores, c=colors, s=sizes, alpha=0.7, edgecolors='black')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add labels for each point
        for i, (date, score, pred, sig) in enumerate(zip(dates, scores, predictions, significances)):
            ax.annotate(f"{pred}\n{sig}", xy=(date, score), xytext=(0, 20),
                        textcoords='offset points', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        # Formatting
        ax.set_title(f'{self.stock_name} - Critical Dates Prediction', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Bullish/Bearish Score', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Rise'),
            Patch(facecolor='red', alpha=0.7, label='Fall'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=10, label='Medium Significance', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=15, label='High Significance', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>♈ Stock Zodiac Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Predict stock trends based on zodiac signs and planetary aspects</p>", unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("Stock Information")
    
    # Stock name input
    stock_name = st.sidebar.text_input("Stock Name", value="AAPL")
    
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
        
        # Predict and display best month
        st.subheader("Best Month Prediction")
        best_month = analysis.predict_best_month()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Best Performing Month")
            st.success(f"**{best_month['month_name']} {best_month['year']}**")
            st.markdown(f"**Score**: {best_month['score']:.1f}")
            
            st.markdown("#### Key Aspects in Best Month")
            for aspect in best_month['aspects']:
                planet = aspect['planet']
                aspect_type = aspect['aspect']
                influence = analysis._get_planet_influence(planet)
                
                if influence == 'Bullish':
                    st.success(f"{planet} {aspect_type}: Bullish")
                elif influence == 'Bearish':
                    st.error(f"{planet} {aspect_type}: Bearish")
                else:
                    st.info(f"{planet} {aspect_type}: Neutral")
        
        with col2:
            st.markdown("#### Other Months Performance")
            # Get all months data
            current_date = datetime.now()
            months_data = []
            for i in range(12):
                target_date = current_date + timedelta(days=30 * i)
                days_diff = (target_date - current_date).days
                
                # Calculate planetary positions for that month
                future_positions = {}
                for planet, angle in analysis.planetary_positions.items():
                    speed = analysis._get_planet_speed(planet)
                    future_angle = (angle + speed * days_diff) % 360
                    future_positions[planet] = future_angle
                
                # Check aspects to the stock's zodiac sign
                future_aspects = analysis._check_aspects_to_sign(analysis.zodiac_angle)
                
                # Calculate score for this month
                month_score = 0
                for aspect in future_aspects:
                    planet = aspect['planet']
                    aspect_type = aspect['aspect']
                    strength = aspect['strength']
                    
                    influence = analysis._get_planet_influence(planet)
                    
                    if influence == 'Bullish' and aspect_type in ['Conjunction', 'Trine', 'Sextile']:
                        month_score += strength
                    elif influence == 'Bearish' and aspect_type in ['Opposition', 'Square']:
                        month_score -= strength
                
                months_data.append({
                    'month': target_date.strftime('%B %Y'),
                    'score': month_score
                })
            
            # Create a DataFrame for display
            months_df = pd.DataFrame(months_data)
            
            # Add performance indicator
            months_df['Performance'] = months_df['score'].apply(
                lambda x: 'Excellent' if x > 5 else ('Good' if x > 0 else 'Poor')
            )
            
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
                # FIXED: Changed 'Date' to 'date' to match the actual column name
                critical_df['date'] = critical_df['date'].dt.strftime('%Y-%m-%d')
                critical_df = critical_df[['date', 'prediction', 'significance', 'score']]
                st.dataframe(critical_df, use_container_width=True)
        else:
            st.info("No critical dates predicted for the next 90 days")
        
        # Display charts
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Planetary Aspects Chart")
            fig1 = analysis.create_planetary_chart()
            st.pyplot(fig1)
        
        with col2:
            st.markdown("#### Trend Prediction Chart")
            fig2 = analysis.create_trend_chart()
            st.pyplot(fig2)
        
        # Display critical dates chart if available
        if critical_dates:
            st.markdown("#### Critical Dates Chart")
            fig3 = analysis.create_critical_dates_chart()
            if fig3:
                st.pyplot(fig3)
        
        # Add disclaimer
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for educational purposes only. Astrological analysis should not be used as the sole basis for trading decisions. Always conduct thorough research and consider multiple factors before making financial decisions.")

if __name__ == "__main__":
    main()
