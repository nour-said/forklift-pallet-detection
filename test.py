
import sys, requests
r = requests.post("http://localhost:8000/predict", files={"file": open(sys.argv[1], "rb")})
print(r.status_code, r.json())