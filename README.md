# Full Scale Body Mass Index Prediction System

This project is a comprehensive Body Mass Index (BMI) prediction system. The system includes multiple deployment methods including serverless and containerized deployments. This README file will guide you through the setup, usage, and deployment of the application.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
  - [Serverless Deployment](#serverless-deployment)
  - [Containerized Deployment](#containerized-deployment)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Full Scale Body Mass Index Prediction System is designed to predict the BMI category based on user inputs such as gender, height, and weight. The system offers flexibility in deployment, allowing you to choose between serverless deployment using AWS Lambda or containerized deployment using Docker and Amazon ECS.

## Features

- **BMI Prediction**: Predicts BMI category based on user inputs.
- **Multiple Deployment Options**: Supports both serverless and containerized deployments.
- **Web Interface**: Provides a web interface for user interaction.
- **Scalable**: Easily scalable using AWS services.

## Prerequisites

- **Python 3.8+**
- **Docker**
- **AWS CLI** configured with your AWS account
- **Zappa** (for serverless deployment)
- **Flask**
- **scikit-learn**
- **pandas**
- **numpy**

## Installation

### Clone the Repository

```bash
git clone https://github.com/bastinjob/Body-Mass-Index-Prediction.git
cd Body-Mass-Index-Prediction


