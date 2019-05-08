# Airbnb_Rental_Price_Prediction
The objective of this project is to model the prices of Airbnb appartments in London.The aim is to build a model to estimate what should be the correct price of their rental given different features and their property.
***


### Problem Statememt:
* The goal of this project is to analyze the Airbnb listings of London and help future users estimate what the correct price of their rental should be given the set of features. The tool will also give simple suggestions to the user to help them get the max rental price while not losing out on the occupancy

### Dataset:
* Data has been scraped from Airbnb by http://insideairbnb.com/get-the-data.html (Download file from here)
* It contains 77000+ records and has 97 columns.

### Project Idea:
1. Deciding the correct price for a short term rental is a complex process. Pricing it too high will result in low occupancy while pricing it too low will result in heavy losses. At present when someone wants to list an Airbnb rental, they have to manually analyze similar properties near their location and decide the price themselves. 
2. The idea of our project is to form a tool which will help the new users estimate what the correct price of their rental should be given the features of their property. We will train the regression model on the available data and then design a console based python program which will ask the users a set of questions regarding their property and then predict what the price of their rental should be.
3. The tool will also provide simple suggestions to the user based on the regression model to help them get the max rental price while not losing out on the occupancy. For example, If the user has set strict cancellation policy, then the tool will provide suggestion like if they change the cancellation policy to moderate they can fetch 10$ additional or say if the user can include internet facility with the room, then it can fetch him an additional 15$.
4. As the dataset has been scraped from Airbnb, it requires extensive cleaning, imputation of the missing values and feature engineering before creating the model. What makes this idea challenging is that no two rentals are the same. All rentals are unique because of their location, amenities, reviews, size, etc. 
***
<p>Thank You!	
<p><!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/shreyaswankhede" aria-label="Follow @shreyaswankhede on GitHub">Follow @shreyaswankhede</a>
