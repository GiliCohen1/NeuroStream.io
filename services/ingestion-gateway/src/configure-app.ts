import { INestApplication, ValidationPipe } from '@nestjs/common';
import { json } from 'express';
import { AllExceptionsFilter } from './common/all-exceptions.filter';

/** Single source of truth for HTTP setup, shared by main.ts and the e2e tests. */
export function configureApp(app: INestApplication): void {
  app.use(json({ limit: '2mb' }));
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }));
  app.useGlobalFilters(new AllExceptionsFilter());
}
