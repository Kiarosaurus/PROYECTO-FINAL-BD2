FROM node:20-alpine

# Next.js necesita esta librería para cargar su binario nativo en alpine
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Instala las dependencias con el lockfile para builds reproducibles
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# La URL del backend se incrusta en el bundle al construir
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
