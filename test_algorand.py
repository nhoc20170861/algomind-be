
from fastapi import APIRouter, HTTPException
from algosdk.v2client import algod

algod_address = "https://testnet-api.4160.nodely.dev"
algod_token = ""

algod_client = algod.AlgodClient(algod_token, algod_address)

router = APIRouter()

@router.get("/algorand/status")
async def get_algorand_status():
    """Get the status of the Algorand node."""
    try:
        status = algod_client.status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node status: {str(e)}")
    
