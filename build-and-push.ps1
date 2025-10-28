# Build and Push Docker Image Script
param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$Version = "v1.0.0"
)

$ImageName = "bankruptcy-prediction"
$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Docker build and push process..." -ForegroundColor Cyan
Write-Host "📦 Image: $DockerHubUsername/$ImageName" -ForegroundColor White

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Build the Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Blue
try {
    docker build -t $ImageName`:latest .
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
    Write-Host "✅ Image built successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to build Docker image: $_" -ForegroundColor Red
    exit 1
}

# Test the image locally
Write-Host "🧪 Testing image locally..." -ForegroundColor Blue
try {
    # Start container in background
    docker run -d -p 8001:8000 --name bankruptcy-test $ImageName`:latest
    
    # Wait for container to start
    Start-Sleep -Seconds 10
    
    # Test health endpoint
    $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Container health check passed!" -ForegroundColor Green
    } else {
        throw "Health check failed"
    }
    
    # Clean up test container
    docker stop bankruptcy-test | Out-Null
    docker rm bankruptcy-test | Out-Null
    
} catch {
    Write-Host "❌ Container test failed: $_" -ForegroundColor Red
    # Clean up on failure
    try {
        docker stop bankruptcy-test 2>$null | Out-Null
        docker rm bankruptcy-test 2>$null | Out-Null
    } catch {}
    exit 1
}

# Login to DockerHub
Write-Host "🔑 Logging into DockerHub..." -ForegroundColor Blue
try {
    docker login
    if ($LASTEXITCODE -ne 0) { throw "DockerHub login failed" }
    Write-Host "✅ Logged into DockerHub successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to login to DockerHub: $_" -ForegroundColor Red
    exit 1
}

# Tag the images
Write-Host "🏷️ Tagging images..." -ForegroundColor Blue
try {
    docker tag $ImageName`:latest $DockerHubUsername/$ImageName`:latest
    docker tag $ImageName`:latest $DockerHubUsername/$ImageName`:$Version
    
    Write-Host "✅ Images tagged successfully!" -ForegroundColor Green
    Write-Host "   📌 $DockerHubUsername/$ImageName`:latest" -ForegroundColor Gray
    Write-Host "   📌 $DockerHubUsername/$ImageName`:$Version" -ForegroundColor Gray
} catch {
    Write-Host "❌ Failed to tag images: $_" -ForegroundColor Red
    exit 1
}

# Push to DockerHub
Write-Host "📤 Pushing to DockerHub..." -ForegroundColor Blue
try {
    Write-Host "   Pushing latest tag..." -ForegroundColor Gray
    docker push $DockerHubUsername/$ImageName`:latest
    if ($LASTEXITCODE -ne 0) { throw "Failed to push latest tag" }
    
    Write-Host "   Pushing version tag..." -ForegroundColor Gray
    docker push $DockerHubUsername/$ImageName`:$Version
    if ($LASTEXITCODE -ne 0) { throw "Failed to push version tag" }
    
    Write-Host "✅ Successfully pushed to DockerHub!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to push to DockerHub: $_" -ForegroundColor Red
    exit 1
}

# Final verification
Write-Host "🔍 Verifying push..." -ForegroundColor Blue
try {
    # Remove local images
    docker rmi $DockerHubUsername/$ImageName`:latest 2>$null | Out-Null
    
    # Pull from DockerHub to verify
    docker pull $DockerHubUsername/$ImageName`:latest | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull from DockerHub" }
    
    Write-Host "✅ Verification successful!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Push completed but verification failed: $_" -ForegroundColor Yellow
}

# Success summary
Write-Host ""
Write-Host "🎉 Docker image successfully built and pushed!" -ForegroundColor Green
Write-Host "🔗 DockerHub URL: https://hub.docker.com/r/$DockerHubUsername/$ImageName" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Usage Commands:" -ForegroundColor White
Write-Host "   docker pull $DockerHubUsername/$ImageName`:latest" -ForegroundColor Gray
Write-Host "   docker run -p 8000:8000 $DockerHubUsername/$ImageName`:latest" -ForegroundColor Gray
Write-Host ""
Write-Host "🚀 Ready for production deployment!" -ForegroundColor Green