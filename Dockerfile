# Multi-stage: publish the .NET app, then copy into the ASP.NET runtime image.
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY *.csproj ./
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /out --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /out ./
# chat session persistence (v1.7.0): SQLite DB lives at /data (PVC-mounted)
USER root
RUN mkdir -p /data && chown 1654:1654 /data
USER 1654:1654
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080 PORT=8080
ENTRYPOINT ["dotnet", "llm-gateway.dll"]
