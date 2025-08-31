# Requirements

The API interface must facilitate retrieving a user's education history. This includes detailed information about each educational institution attended by the user, such as the school name, major, degree, achievements, and the start and end dates of attendance. The API must handle errors and validate input parameters effectively.

# Protocol Flow

## Interaction Flow

1. The client sends a request to the server with the user's ID and an optional parameter to include detailed information.
2. The server processes the request, validates the input parameters, and retrieves the education history for the specified user.
3. The server sends back a response containing the user's education history or an error message.

# Data Format

## Request Message Format

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EducationHistoryRequest",
  "type": "object",
  "properties": {
    "messageType": {
      "type": "string",
      "const": "EducationHistoryRequest"
    },
    "messageId": {
      "type": "string",
      "description": "A unique identifier for the request message"
    },
    "userId": {
      "type": "string",
      "description": "The unique identifier for the user",
      "minLength": 1
    },
    "includeDetails": {
      "type": "boolean",
      "description": "Optional flag to include detailed information",
      "default": false
    }
  },
  "required": ["messageType", "messageId", "userId"],
  "additionalProperties": false
}
```

## Response Message Format

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EducationHistoryResponse",
  "type": "object",
  "properties": {
    "messageType": {
      "type": "string",
      "const": "EducationHistoryResponse"
    },
    "messageId": {
      "type": "string",
      "description": "The unique identifier matching the request message"
    },
    "code": {
      "type": "integer",
      "description": "HTTP status code indicating the result of the request"
    },
    "educationHistory": {
      "type": "array",
      "description": "List of education history records",
      "items": {
        "type": "object",
        "properties": {
          "institution": {
            "type": "string",
            "description": "Name of the educational institution"
          },
          "major": {
            "type": "string",
            "description": "Major studied at the institution"
          },
          "degree": {
            "type": "string",
            "description": "Degree achieved",
            "enum": ["Bachelor", "Master", "Doctorate"]
          },
          "achievements": {
            "type": "string",
            "description": "Achievements while attending the institution"
          },
          "startDate": {
            "type": "string",
            "format": "date",
            "description": "Start date of attendance in YYYY-MM-DD format"
          },
          "endDate": {
            "type": "string",
            "format": "date",
            "description": "End date of attendance in YYYY-MM-DD format"
          }
        },
        "required": ["institution", "major", "degree", "startDate", "endDate"],
        "additionalProperties": false
      }
    }
  },
  "required": ["messageType", "messageId", "code", "educationHistory"],
  "additionalProperties": false
}
```

# Error Handling

- **HTTP Status Codes**:
  - **200 OK**: Successful retrieval of education history.
  - **400 Bad Request**: Invalid input parameters, e.g., missing or incorrectly formatted `userId`.
  - **404 Not Found**: User ID does not exist in the system.
  - **500 Internal Server Error**: An unexpected error occurred on the server.

- **Error Response Example**:

```json
{
  "messageType": "EducationHistoryResponse",
  "messageId": "abcd-1234",
  "code": 400,
  "error": {
    "errorCode": "INVALID_USER_ID",
    "errorDescription": "The provided user ID is invalid or not found."
  }
}
```

This protocol ensures efficient retrieval of a user's educational background while providing comprehensive error handling and detailed validation of input parameters.