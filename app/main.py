from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from app.inference import EVMaintenancePredictor
from app.schemas import (
    VehicleData, 
    PredictionResponse, 
    HealthResponse,
    ErrorResponse
)
from app.config import MODEL_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor_instance = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    logger.info(" Starting up API server...")
    try:
        predictor = EVMaintenancePredictor()
        predictor_instance["predictor"] = predictor
        logger.info("Models loaded ")
    except Exception as e:
        logger.error(f"Failed to load models")
        raise
    
    yield  
    
    logger.info(" Shutting down")
    predictor_instance.clear()
    logger.info(" Cleanup complete")

app = FastAPI(
    title="EV Predictive Maintenance API",
    description="ML-powered battery health monitoring and maintenance prediction",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ev-fleet-predictive-maintenance-backend.onrender.com/openapi.json"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict battery health and maintenance needs"
)
async def predict_maintenance(vehicle_data: VehicleData):
   
    try:
        predictor = predictor_instance.get("predictor")
        
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Server may be starting up."
            )
        
        
        vehicle_dict = vehicle_data.model_dump()

        logger.info(f"Processing prediction for {vehicle_dict.get('Vehicle_ID')}")
        result = predictor.predict(vehicle_dict)
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get('error_message')
            )
        
        logger.info(f"Prediction successful for {vehicle_dict.get('Vehicle_ID')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "EV Predictive Maintenance",
        "version": MODEL_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict", 
            "model_info": "GET /model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat()
    }
@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information"
)
async def get_model_info():
   
    predictor = predictor_instance.get("predictor")
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    return {
        "model_version": MODEL_VERSION,
        "features": predictor.feature_columns,
        "models": {
            "soh_predictor": str(type(predictor.soh_model).__name__),
            "thermal_predictor": str(type(predictor.thermal_model).__name__)
        },
        "thresholds": {
            "soh_critical": 0.60,
            "soh_warning": 0.80,
            "rul_urgent": 100,
            "thermal_danger": 0.70
        }
    }

